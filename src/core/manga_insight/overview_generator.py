"""
全书概览生成器

从层级分析结果生成全书概述。
"""

import logging
from typing import Dict, Callable, Awaitable
from datetime import datetime

from .config_models import MangaInsightConfig
from .storage import AnalysisStorage
from .config_utils import create_chat_client

logger = logging.getLogger("MangaInsight.OverviewGenerator")


class OverviewGenerator:
    """
    全书概览生成器

    负责生成全书概述，使用层级式摘要方法。
    """

    def __init__(
        self,
        book_id: str,
        config: MangaInsightConfig,
        storage: AnalysisStorage,
        get_book_info_func: Callable[[], Awaitable[Dict]]
    ):
        """
        初始化概览生成器

        Args:
            book_id: 书籍ID
            config: 配置
            storage: 存储
            get_book_info_func: 获取书籍信息的异步函数
        """
        self.book_id = book_id
        self.config = config
        self.storage = storage
        self._get_book_info = get_book_info_func

    async def generate_overview(self) -> Dict:
        """生成全书概述（层级式摘要）"""
        from .features.hierarchical_summary import HierarchicalSummaryGenerator

        book_info = await self._get_book_info()
        chapters = await self.storage.list_chapters()

        # 创建 LLM 客户端（使用工厂函数）
        llm_client = None
        try:
            llm_client = create_chat_client(self.config)
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败，将使用简单合并: {e}")

        # 使用层级式摘要生成器
        summary_generator = HierarchicalSummaryGenerator(
            book_id=self.book_id,
            storage=self.storage,
            llm_client=llm_client,
            book_info=book_info,
            prompts_config=self.config.prompts
        )

        summary_source = "none"
        book_summary = ""
        section_summaries = []

        try:
            # 生成层级概述
            hierarchical_result = await summary_generator.generate_hierarchical_overview()
            book_summary = hierarchical_result.get("book_summary", "")
            section_summaries = hierarchical_result.get("section_summaries", [])
            summary_source = hierarchical_result.get("source", "unknown")
            logger.info(f"概要生成完成，数据来源: {summary_source}")

            # 自动生成无剧透简介模板
            try:
                await summary_generator.generate_with_template("no_spoiler")
                logger.info(f"无剧透简介模板生成完成")
            except Exception as e:
                logger.warning(f"无剧透简介模板生成失败: {e}")

        except Exception as e:
            logger.error(f"层级摘要生成失败: {e}", exc_info=True)
            book_summary = "概要生成失败，请重试。"
            section_summaries = []
        finally:
            if llm_client:
                try:
                    await llm_client.close()
                except Exception:
                    pass

        overview = {
            "book_id": self.book_id,
            "title": book_info.get("title", ""),
            "total_pages": book_info.get("total_pages", 0),
            "total_chapters": len(chapters),
            "summary": book_summary,
            "section_summaries": section_summaries,
            "summary_source": summary_source,
            "themes": [],
            "generated_at": datetime.now().isoformat()
        }

        await self.storage.save_overview(overview)
        return overview

    async def regenerate_with_template(self, template_id: str) -> Dict:
        """使用指定模板重新生成概述"""
        from .features.hierarchical_summary import HierarchicalSummaryGenerator

        book_info = await self._get_book_info()

        llm_client = None
        try:
            llm_client = create_chat_client(self.config)

            summary_generator = HierarchicalSummaryGenerator(
                book_id=self.book_id,
                storage=self.storage,
                llm_client=llm_client,
                book_info=book_info,
                prompts_config=self.config.prompts
            )

            result = await summary_generator.generate_with_template(template_id)
            return result

        except Exception as e:
            logger.error(f"模板概述生成失败: {e}", exc_info=True)
            raise
        finally:
            if llm_client:
                try:
                    await llm_client.close()
                except Exception:
                    pass
