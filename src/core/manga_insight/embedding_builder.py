# src/core/manga_insight/embedding_builder.py
"""
向量嵌入构建器模块

从 analyzer.py 拆分，负责向量嵌入的构建。
"""

import logging
from typing import Dict, List

from tqdm import tqdm

from .storage import AnalysisStorage
from .embedding_client import EmbeddingClient
from .vector_store import MangaVectorStore

logger = logging.getLogger("MangaInsight.EmbeddingBuilder")


class EmbeddingBuilder:
    """
    向量嵌入构建器

    负责构建页面和事件的向量嵌入。
    """

    def __init__(
        self,
        book_id: str,
        storage: AnalysisStorage,
        embedding: EmbeddingClient,
        vector_store: MangaVectorStore
    ):
        self.book_id = book_id
        self.storage = storage
        self.embedding = embedding
        self.vector_store = vector_store

    async def build_embeddings(self) -> Dict:
        """
        构建向量嵌入（页面 + 事件）

        基于批量分析结果构建两层索引：
        1. 页面级向量 (page_summary)
        2. 事件级向量 (key_events) - 细粒度检索

        Returns:
            Dict: 构建结果统计
        """
        if not self.embedding or not self.vector_store.is_available():
            logger.warning("向量功能不可用，跳过构建嵌入")
            return {"success": False, "error": "向量功能不可用"}

        # 获取所有批次
        batches = await self.storage.list_batches()

        if not batches:
            # 降级：从页面分析构建
            logger.info("无批次数据，从页面分析构建向量")
            return await self._build_embeddings_from_pages()

        logger.info(f"开始构建向量嵌入: 共 {len(batches)} 个批次")

        # 清除现有向量
        await self.vector_store.delete_all_pages()
        await self.vector_store.delete_all_events()

        pages_count = 0
        events_count = 0
        skip_count = 0

        for batch_info in tqdm(batches, desc="构建向量嵌入", unit="批次"):
            start_page = batch_info["start_page"]
            end_page = batch_info["end_page"]
            batch_id = f"batch_{start_page}_{end_page}"

            batch_data = await self.storage.load_batch_analysis(start_page, end_page)
            if not batch_data:
                skip_count += 1
                continue

            # 1. 页面级向量
            for page in batch_data.get("pages", []):
                page_num = page.get("page_number")
                page_summary = page.get("page_summary", "")

                if page_num and page_summary:
                    try:
                        embedding = await self.embedding.embed(page_summary)
                        await self.vector_store.add_page_embedding(
                            page_num=page_num,
                            embedding=embedding,
                            metadata={
                                "page_summary": page_summary,
                                "type": "page",
                                "parent_batch": batch_id
                            }
                        )
                        pages_count += 1
                    except Exception as e:
                        logger.warning(f"页面 {page_num} 向量化失败: {e}")

            # 2. 事件级向量
            key_events = batch_data.get("key_events", [])
            for event_idx, event in enumerate(key_events):
                if not event or not isinstance(event, str):
                    continue
                event = event.strip()
                if len(event) < 5:
                    continue

                try:
                    embedding = await self.embedding.embed(event)
                    event_id = f"event_{start_page}_{end_page}_{event_idx}"

                    await self.vector_store.add_event_embedding(
                        event_id=event_id,
                        embedding=embedding,
                        metadata={
                            "content": event,
                            "type": "event",
                            "parent_batch": batch_id,
                            "start_page": start_page,
                            "end_page": end_page
                        }
                    )
                    events_count += 1
                except Exception as e:
                    logger.warning(f"事件向量化失败 ({batch_id}): {e}")

        result = {
            "success": True,
            "pages_count": pages_count,
            "events_count": events_count,
            "total_count": pages_count + events_count,
            "batches_processed": len(batches) - skip_count,
            "batches_skipped": skip_count
        }

        logger.info(
            f"向量嵌入构建完成: {pages_count} 页面, {events_count} 事件, "
            f"共 {pages_count + events_count} 条向量"
        )
        return result

    async def _build_embeddings_from_pages(self) -> Dict:
        """从页面分析构建向量（降级方案）"""
        page_nums = await self.storage.list_pages()

        if not page_nums:
            logger.warning("没有已分析的页面，跳过构建嵌入")
            return {"success": False, "error": "没有已分析的页面"}

        logger.info(f"从页面分析构建向量: 共 {len(page_nums)} 页")

        # 清除现有向量
        await self.vector_store.delete_all_pages()

        pages_count = 0
        skip_count = 0

        for page_num in tqdm(page_nums, desc="构建向量嵌入", unit="页"):
            analysis = await self.storage.load_page_analysis(page_num)
            if not analysis:
                skip_count += 1
                continue

            summary = analysis.get("page_summary", "")
            if summary:
                try:
                    embedding = await self.embedding.embed(summary)
                    await self.vector_store.add_page_embedding(
                        page_num, embedding, {
                            "page_summary": summary,
                            "type": "page"
                        }
                    )
                    pages_count += 1
                except Exception as e:
                    logger.warning(f"第 {page_num} 页向量化失败: {e}")
                    skip_count += 1
            else:
                skip_count += 1

        result = {
            "success": pages_count > 0,
            "pages_count": pages_count,
            "events_count": 0,
            "total_count": pages_count,
            "skip_count": skip_count
        }

        logger.info(f"向量嵌入构建完成: 成功 {pages_count} 页, 跳过 {skip_count} 页")
        return result
