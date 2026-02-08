# src/core/manga_insight/summary_generator.py
"""
摘要生成器模块

从 analyzer.py 拆分，负责段落和章节摘要生成。
"""

import logging
from typing import Dict, List
from datetime import datetime

from .storage import AnalysisStorage
from .config_models import (
    MangaInsightConfig,
    DEFAULT_SEGMENT_SUMMARY_PROMPT,
    DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT,
    DEFAULT_ANALYSIS_SYSTEM_PROMPT
)
from .utils.json_parser import parse_llm_json
from .config_utils import create_chat_client

logger = logging.getLogger("MangaInsight.SummaryGenerator")


class SummaryGenerator:
    """
    摘要生成器

    负责段落摘要和章节摘要的生成。
    """

    def __init__(self, book_id: str, config: MangaInsightConfig, storage: AnalysisStorage):
        self.book_id = book_id
        self.config = config
        self.storage = storage

    async def generate_segment_summary(
        self,
        segment_id: str,
        batch_results: List[Dict],
        force: bool = False
    ) -> Dict:
        """
        生成小总结（第二层级）

        Args:
            segment_id: 小总结ID
            batch_results: 批量分析结果列表
            force: 是否强制重新生成

        Returns:
            Dict: 小总结
        """
        if not batch_results:
            return {}

        # 检查缓存
        if not force:
            cached = await self.storage.load_segment_summary(segment_id)
            if cached:
                logger.debug(f"使用小总结缓存: {segment_id}")
                return cached

        # 收集批量摘要
        batch_summaries = []
        all_key_events = []
        start_page = float('inf')
        end_page = 0

        for batch in batch_results:
            page_range = batch.get("page_range", {})
            batch_start = page_range.get("start", 0)
            batch_end = page_range.get("end", 0)

            start_page = min(start_page, batch_start)
            end_page = max(end_page, batch_end)

            summary = batch.get("batch_summary", "")
            if summary:
                batch_summaries.append(f"第{batch_start}-{batch_end}页: {summary}")

            events = batch.get("key_events", [])
            all_key_events.extend(events)

        # 使用 LLM 生成小总结
        from .embedding_client import ChatClient

        segment_data = {
            "segment_id": segment_id,
            "page_range": {"start": int(start_page), "end": int(end_page)},
            "summary": "",
            "key_events": all_key_events[:10],
            "plot_progression": "",
            "themes": [],
            "batch_count": len(batch_results)
        }

        llm = None
        try:
            llm = create_chat_client(self.config)

            # 构建提示词
            batch_summaries_text = "\n".join(batch_summaries)
            base_prompt = self.config.prompts.segment_summary if self.config.prompts.segment_summary else DEFAULT_SEGMENT_SUMMARY_PROMPT
            prompt = base_prompt.replace("{batch_count}", str(len(batch_results)))
            prompt = prompt.replace("{start_page}", str(int(start_page)))
            prompt = prompt.replace("{end_page}", str(int(end_page)))
            prompt = prompt.replace("{batch_summaries}", batch_summaries_text)
            prompt = prompt.replace("{segment_id}", str(segment_id))

            system_prompt = self.config.prompts.analysis_system if self.config.prompts.analysis_system else DEFAULT_ANALYSIS_SYSTEM_PROMPT
            response = await llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3
            )

            # 解析 JSON 响应
            parsed = parse_llm_json(response)
            if parsed:
                segment_data.update(parsed)
            else:
                segment_data["summary"] = "\n".join(batch_summaries[:3])

        except Exception as e:
            logger.warning(f"小总结生成失败: {e}")
            segment_data["summary"] = "\n".join(batch_summaries[:3])
        finally:
            if llm:
                await llm.close()

        # 保存小总结
        segment_data["generated_at"] = datetime.now().isoformat()
        await self.storage.save_segment_summary(segment_id, segment_data)

        return segment_data

    async def generate_chapter_from_segments(
        self,
        chapter_id: str,
        chapter_info: Dict,
        segment_results: List[Dict]
    ) -> Dict:
        """从小总结生成章节总结"""
        from .embedding_client import ChatClient

        if not segment_results:
            return {
                "chapter_id": chapter_id,
                "title": chapter_info.get("title", f"第{chapter_id}章"),
                "summary": "",
                "analyzed_at": datetime.now().isoformat()
            }

        # 收集小总结信息
        segment_summaries = []
        all_events = []
        start_page = float('inf')
        end_page = 0

        for seg in segment_results:
            page_range = seg.get("page_range", {})
            start_page = min(start_page, page_range.get("start", 0))
            end_page = max(end_page, page_range.get("end", 0))

            summary = seg.get("summary", "")
            if summary:
                seg_range = f"第{page_range.get('start')}-{page_range.get('end')}页"
                segment_summaries.append(f"{seg_range}: {summary}")

            all_events.extend(seg.get("key_events", []))

        # 使用 LLM 生成章节总结
        chapter_summary = ""
        llm = None
        try:
            llm = create_chat_client(self.config)

            segment_summaries_text = "\n".join(segment_summaries)
            base_prompt = self.config.prompts.chapter_summary if self.config.prompts.chapter_summary else DEFAULT_CHAPTER_FROM_SEGMENTS_PROMPT
            prompt = base_prompt.replace("{chapter_title}", chapter_info.get("title", f"第{chapter_id}章"))
            prompt = prompt.replace("{chapter_id}", str(chapter_id))
            prompt = prompt.replace("{start_page}", str(int(start_page)))
            prompt = prompt.replace("{end_page}", str(int(end_page)))
            prompt = prompt.replace("{segment_summaries}", segment_summaries_text)

            system_prompt = self.config.prompts.analysis_system if self.config.prompts.analysis_system else DEFAULT_ANALYSIS_SYSTEM_PROMPT
            response = await llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3
            )

            # 解析 JSON
            parsed = parse_llm_json(response)
            if parsed:
                chapter_summary = parsed.get("summary", "")

                return {
                    "chapter_id": chapter_id,
                    "title": chapter_info.get("title", f"第{chapter_id}章"),
                    "page_range": {"start": int(start_page), "end": int(end_page)},
                    "analyzed_at": datetime.now().isoformat(),
                    "analysis_mode": "four_tier",
                    "summary": chapter_summary,
                    "main_plot": parsed.get("main_plot", ""),
                    "plot_events": parsed.get("key_events", all_events[:10]),
                    "themes": parsed.get("themes", []),
                    "atmosphere": parsed.get("atmosphere", ""),
                    "connections": parsed.get("connections", {}),
                    "segment_count": len(segment_results)
                }
            else:
                chapter_summary = response.strip()

        except Exception as e:
            logger.warning(f"章节总结生成失败: {e}")
            chapter_summary = "\n".join(segment_summaries[:3])
        finally:
            if llm:
                await llm.close()

        return {
            "chapter_id": chapter_id,
            "title": chapter_info.get("title", f"第{chapter_id}章"),
            "page_range": {"start": int(start_page), "end": int(end_page)},
            "analyzed_at": datetime.now().isoformat(),
            "analysis_mode": "four_tier",
            "summary": chapter_summary,
            "plot_events": all_events[:10],
            "segment_count": len(segment_results)
        }
