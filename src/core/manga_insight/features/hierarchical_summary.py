"""
Manga Insight 层级式摘要生成器（重构版）

统一的概要生成逻辑：
1. 优先使用分析过程产生的数据（segment/chapter 总结）
2. 降级方案：从页面摘要自动分组生成
3. 所有提示词从配置中读取
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..config_models import (
    PromptsConfig, 
    DEFAULT_GROUP_SUMMARY_PROMPT, 
    DEFAULT_BOOK_OVERVIEW_PROMPT
)

logger = logging.getLogger("MangaInsight.HierarchicalSummary")


class HierarchicalSummaryGenerator:
    """层级式摘要生成器（重构版）"""
    
    PAGES_PER_GROUP = 10  # 每组页数（降级方案使用）
    
    def __init__(
        self, 
        book_id: str, 
        storage, 
        llm_client=None, 
        book_info: Dict = None,
        prompts_config: PromptsConfig = None
    ):
        """
        Args:
            book_id: 书籍ID
            storage: AnalysisStorage 实例
            llm_client: LLM客户端
            book_info: 书籍信息（包含用户创建的章节）
            prompts_config: 提示词配置
        """
        self.book_id = book_id
        self.storage = storage
        self.llm = llm_client
        self.book_info = book_info or {}
        self.prompts = prompts_config or PromptsConfig()
    
    async def generate_hierarchical_overview(self) -> Dict:
        """
        生成层级式概述
        
        优先级：
        1. 从章节总结生成（如果有 chapter 分析结果）
        2. 从段落总结生成（如果有 segment 分析结果）
        3. 从批量分析生成（如果有 batch 分析结果）
        4. 从页面摘要生成（降级方案）
        
        Returns:
            Dict: 包含 book_summary 和 section_summaries 的概述
        """
        logger.info(f"开始生成层级概述: {self.book_id}")
        
        # 方案1：从章节总结生成
        chapters = await self.storage.list_chapters()
        if chapters:
            logger.info(f"使用方案1：从 {len(chapters)} 个章节总结生成概要")
            return await self._generate_from_chapters(chapters)
        
        # 方案2：从段落总结生成
        segments = await self.storage.list_segments()
        if segments:
            logger.info(f"使用方案2：从 {len(segments)} 个段落总结生成概要")
            return await self._generate_from_segments(segments)
        
        # 方案3：从批量分析生成
        batches = await self.storage.list_batches()
        if batches:
            logger.info(f"使用方案3：从 {len(batches)} 个批量分析生成概要")
            return await self._generate_from_batches(batches)
        
        # 方案4：从页面摘要生成（降级）
        page_nums = await self.storage.list_pages()
        if page_nums:
            logger.info(f"使用方案4（降级）：从 {len(page_nums)} 个页面摘要生成概要")
            return await self._generate_from_pages(page_nums)
        
        # 无数据
        logger.warning("没有可用的分析数据")
        return self._empty_result()
    
    async def _generate_from_chapters(self, chapters: List[Dict]) -> Dict:
        """从章节总结生成概要"""
        chapter_texts = []
        section_summaries = []
        
        for ch in chapters:
            ch_id = ch.get("id", "")
            analysis = await self.storage.load_chapter_analysis(ch_id)
            if analysis:
                title = analysis.get("title", ch_id)
                summary = analysis.get("summary", "")
                page_range = analysis.get("page_range", {})
                
                chapter_texts.append(f"【{title}】{summary}")
                section_summaries.append({
                    "chapter_id": ch_id,
                    "chapter_title": title,
                    "start_page": page_range.get("start", 0),
                    "end_page": page_range.get("end", 0),
                    "summary": summary
                })
        
        if not chapter_texts:
            return self._empty_result()
        
        # 生成全书概要
        book_summary = await self._generate_book_summary("\n\n".join(chapter_texts))
        
        return {
            "book_summary": book_summary,
            "section_summaries": section_summaries,
            "source": "chapters",
            "group_count": len(chapters),
            "total_pages": self._get_total_pages(section_summaries),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_from_segments(self, segments: List[Dict]) -> Dict:
        """从段落总结生成概要"""
        segment_texts = []
        section_summaries = []
        
        for seg in segments:
            seg_id = seg.get("segment_id", "")
            full_data = await self.storage.load_segment_summary(seg_id)
            if full_data:
                summary = full_data.get("summary", "")
                page_range = full_data.get("page_range", {})
                start = page_range.get("start", 0)
                end = page_range.get("end", 0)
                
                segment_texts.append(f"第{start}-{end}页: {summary}")
                section_summaries.append({
                    "segment_id": seg_id,
                    "start_page": start,
                    "end_page": end,
                    "summary": summary
                })
        
        if not segment_texts:
            return self._empty_result()
        
        # 生成全书概要
        book_summary = await self._generate_book_summary("\n\n".join(segment_texts))
        
        return {
            "book_summary": book_summary,
            "section_summaries": section_summaries,
            "source": "segments",
            "group_count": len(segments),
            "total_pages": self._get_total_pages(section_summaries),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_from_batches(self, batches: List[Dict]) -> Dict:
        """从批量分析生成概要"""
        batch_texts = []
        section_summaries = []
        
        for batch_info in batches:
            start = batch_info.get("start_page", 0)
            end = batch_info.get("end_page", 0)
            batch_data = await self.storage.load_batch_analysis(start, end)
            if batch_data:
                summary = batch_data.get("batch_summary", "")
                batch_texts.append(f"第{start}-{end}页: {summary}")
                section_summaries.append({
                    "start_page": start,
                    "end_page": end,
                    "summary": summary
                })
        
        if not batch_texts:
            return self._empty_result()
        
        # 如果批次过多，先合并
        if len(section_summaries) > 10:
            section_summaries = await self._merge_sections(section_summaries)
            batch_texts = [
                f"第{s['start_page']}-{s['end_page']}页: {s['summary']}" 
                for s in section_summaries
            ]
        
        book_summary = await self._generate_book_summary("\n\n".join(batch_texts))
        
        return {
            "book_summary": book_summary,
            "section_summaries": section_summaries,
            "source": "batches",
            "group_count": len(section_summaries),
            "total_pages": self._get_total_pages(section_summaries),
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_from_pages(self, page_nums: List[int]) -> Dict:
        """从页面摘要生成概要（降级方案）"""
        # 收集页面摘要
        page_summaries = {}
        for page_num in page_nums:
            analysis = await self.storage.load_page_analysis(page_num)
            if analysis and analysis.get("page_summary"):
                page_summaries[page_num] = analysis.get("page_summary")
        
        if not page_summaries:
            return self._empty_result()
        
        total_pages = max(page_summaries.keys())
        page_count = len(page_summaries)
        
        # 根据页数选择策略
        if page_count <= 5:
            # 少量页面：直接生成
            sorted_pages = sorted(page_summaries.keys())
            texts = [f"第{p}页: {page_summaries[p]}" for p in sorted_pages]
            book_summary = await self._generate_book_summary("\n".join(texts))
            return {
                "book_summary": book_summary,
                "section_summaries": [],
                "source": "pages_direct",
                "group_count": 1,
                "total_pages": total_pages,
                "generated_at": datetime.now().isoformat()
            }
        
        # 分组生成
        section_summaries = await self._group_pages_and_summarize(page_summaries)
        
        # 如果分组过多，再合并一层
        if len(section_summaries) > 10:
            section_summaries = await self._merge_sections(section_summaries)
        
        section_texts = [
            f"第{s['start_page']}-{s['end_page']}页: {s['summary']}" 
            for s in section_summaries
        ]
        book_summary = await self._generate_book_summary("\n\n".join(section_texts))
        
        return {
            "book_summary": book_summary,
            "section_summaries": section_summaries,
            "source": "pages_grouped",
            "group_count": len(section_summaries),
            "total_pages": total_pages,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _group_pages_and_summarize(self, page_summaries: Dict[int, str]) -> List[Dict]:
        """将页面分组并生成摘要"""
        sorted_pages = sorted(page_summaries.keys())
        sections = []
        
        for i in range(0, len(sorted_pages), self.PAGES_PER_GROUP):
            group_pages = sorted_pages[i:i + self.PAGES_PER_GROUP]
            group_texts = [f"第{p}页: {page_summaries[p]}" for p in group_pages]
            
            start_page = group_pages[0]
            end_page = group_pages[-1]
            
            # 使用配置的提示词
            if self.llm:
                prompt = self.prompts.group_summary or DEFAULT_GROUP_SUMMARY_PROMPT
                prompt = prompt.format(
                    start_page=start_page,
                    end_page=end_page,
                    page_contents="\n".join(group_texts)
                )
                summary = await self._call_llm(prompt)
            else:
                summary = self._simple_merge(group_texts)
            
            sections.append({
                "start_page": start_page,
                "end_page": end_page,
                "summary": summary
            })
        
        return sections
    
    async def _merge_sections(self, sections: List[Dict], merge_count: int = 5) -> List[Dict]:
        """合并多个分段为更大的块"""
        merged = []
        
        for i in range(0, len(sections), merge_count):
            group = sections[i:i + merge_count]
            texts = [f"第{s['start_page']}-{s['end_page']}页: {s['summary']}" for s in group]
            
            start_page = group[0]["start_page"]
            end_page = group[-1]["end_page"]
            
            if self.llm:
                prompt = self.prompts.group_summary or DEFAULT_GROUP_SUMMARY_PROMPT
                prompt = prompt.format(
                    start_page=start_page,
                    end_page=end_page,
                    page_contents="\n".join(texts)
                )
                summary = await self._call_llm(prompt)
            else:
                summary = self._simple_merge(texts, max_chars=400)
            
            merged.append({
                "start_page": start_page,
                "end_page": end_page,
                "summary": summary
            })
        
        return merged
    
    async def _generate_book_summary(self, content: str) -> str:
        """生成全书概要"""
        if not self.llm:
            return content[:500] + "..." if len(content) > 500 else content
        
        prompt = self.prompts.book_overview or DEFAULT_BOOK_OVERVIEW_PROMPT
        prompt = prompt.format(section_summaries=content)
        
        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return prompt[:300] + "..." if len(prompt) > 300 else prompt
    
    def _simple_merge(self, texts: List[str], max_chars: int = 200) -> str:
        """简单合并（无 LLM 时的降级方案）"""
        merged = " ".join(texts)
        if len(merged) > max_chars:
            return merged[:max_chars] + "..."
        return merged
    
    def _get_total_pages(self, sections: List[Dict]) -> int:
        """获取总页数"""
        if not sections:
            return 0
        return max(s.get("end_page", 0) for s in sections)
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "book_summary": "暂无概要，请完成分析后查看。",
            "section_summaries": [],
            "source": "none",
            "group_count": 0,
            "total_pages": 0,
            "generated_at": datetime.now().isoformat()
        }
