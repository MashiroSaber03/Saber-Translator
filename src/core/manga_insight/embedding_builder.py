# src/core/manga_insight/embedding_builder.py
"""
向量嵌入构建器模块

从 analyzer.py 拆分，负责向量嵌入的构建。
"""

import logging
import threading
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from .storage import AnalysisStorage
from .embedding_client import EmbeddingClient
from .vector_store import MangaVectorStore

logger = logging.getLogger("MangaInsight.EmbeddingBuilder")
_build_locks: Dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def _get_build_lock(book_id: str) -> threading.Lock:
    with _build_locks_guard:
        lock = _build_locks.get(book_id)
        if lock is None:
            lock = threading.Lock()
            _build_locks[book_id] = lock
        return lock


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

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[Callable[[int, int, str, str], None]],
        current: int,
        total: int,
        phase: str,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(current, total, phase, message)
        except Exception as exc:
            logger.debug("向量重建进度回调失败: %s", exc)

    async def build_embeddings(
        self,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> Dict:
        """
        构建向量嵌入（页面 + 事件）

        基于批量分析结果构建两层索引：
        1. 页面级向量 (page_summary)
        2. 事件级向量 (key_events) - 细粒度检索

        若 batches/ 不完整（部分批次丢失），会从 pages/ 单页分析补齐缺失页面，
        避免出现"batches 仅 4 个 → 只处理 20 页 → pages/ 里 1100+ 页被无视"的数据丢失。

        Returns:
            Dict: 构建结果统计
        """
        build_lock = _get_build_lock(self.book_id)
        if not build_lock.acquire(blocking=False):
            logger.warning("书籍 %s 的向量重建已在进行中，拒绝重入", self.book_id)
            return {
                "success": False,
                "error": "该书籍的向量重建正在进行中，请稍后再试",
                "error_code": "EMBEDDING_REBUILD_IN_PROGRESS",
                "status_code": 409,
            }

        try:
            if not self.embedding or not self.vector_store.is_available():
                logger.warning("向量功能不可用，跳过构建嵌入")
                return {"success": False, "error": "向量功能不可用"}

            # 获取所有批次
            batches = await self.storage.list_batches()

            if not batches:
                # 完全没有 batches，直接走 pages/ 兜底
                logger.info("无批次数据，从页面分析构建向量")
                return await self._build_embeddings_from_pages(progress_callback=progress_callback)

            logger.info(f"开始构建向量嵌入: 共 {len(batches)} 个批次")
            self._emit_progress(progress_callback, 0, len(batches), "embedding_batches", f"重建批次向量 0/{len(batches)}")

            # 重建前彻底重置 collections，避免旧 collection 保留旧维度元数据
            if not await self.vector_store.delete_all():
                logger.error("重置向量 collections 失败，终止重建")
                return {"success": False, "error": "重置向量库失败"}

            pages_count = 0
            events_count = 0
            skip_count = 0
            covered_pages: set = set()

            for batch_index, batch_info in enumerate(tqdm(batches, desc="构建向量嵌入", unit="批次"), start=1):
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
                            added = await self.vector_store.add_page_embedding(
                                page_num=page_num,
                                embedding=embedding,
                                metadata={
                                    "page_summary": page_summary,
                                    "type": "page",
                                    "parent_batch": batch_id
                                }
                            )
                            if added:
                                pages_count += 1
                                covered_pages.add(page_num)
                            else:
                                logger.warning(f"页面 {page_num} 向量写入失败")
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

                        added = await self.vector_store.add_event_embedding(
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
                        if added:
                            events_count += 1
                        else:
                            logger.warning(f"事件向量写入失败 ({event_id})")
                    except Exception as e:
                        logger.warning(f"事件向量化失败 ({batch_id}): {e}")

                self._emit_progress(
                    progress_callback,
                    batch_index,
                    len(batches),
                    "embedding_batches",
                    f"重建批次向量 {batch_index}/{len(batches)}",
                )

            # 用 pages/ 单页分析补齐 batches 未覆盖的页面
            all_page_nums = await self.storage.list_pages()
            uncovered = [p for p in all_page_nums if p not in covered_pages]
            filled_count = 0

            if uncovered:
                logger.info(
                    f"批次覆盖 {len(covered_pages)} 页，从单页分析补齐剩余 {len(uncovered)} 页"
                )
                self._emit_progress(progress_callback, 0, len(uncovered), "embedding_fallback_pages", f"补齐页面向量 0/{len(uncovered)}")
                for uncovered_index, page_num in enumerate(tqdm(uncovered, desc="补齐页面向量", unit="页"), start=1):
                    analysis = await self.storage.load_page_analysis(page_num)
                    if not analysis:
                        self._emit_progress(progress_callback, uncovered_index, len(uncovered), "embedding_fallback_pages", f"补齐页面向量 {uncovered_index}/{len(uncovered)}")
                        continue
                    summary = analysis.get("page_summary", "")
                    if not summary:
                        self._emit_progress(progress_callback, uncovered_index, len(uncovered), "embedding_fallback_pages", f"补齐页面向量 {uncovered_index}/{len(uncovered)}")
                        continue
                    try:
                        embedding = await self.embedding.embed(summary)
                        added = await self.vector_store.add_page_embedding(
                            page_num=page_num,
                            embedding=embedding,
                            metadata={
                                "page_summary": summary,
                                "type": "page",
                                "parent_batch": "page_fallback"
                            }
                        )
                        if added:
                            pages_count += 1
                            filled_count += 1
                        else:
                            logger.warning(f"补齐页面 {page_num} 向量写入失败")
                    except Exception as e:
                        logger.warning(f"补齐页面 {page_num} 向量化失败: {e}")
                    self._emit_progress(progress_callback, uncovered_index, len(uncovered), "embedding_fallback_pages", f"补齐页面向量 {uncovered_index}/{len(uncovered)}")

            result = {
                "success": True,
                "pages_count": pages_count,
                "events_count": events_count,
                "total_count": pages_count + events_count,
                "batches_processed": len(batches) - skip_count,
                "batches_skipped": skip_count,
                "pages_filled_from_fallback": filled_count,
            }

            logger.info(
                f"向量嵌入构建完成: {pages_count} 页面 (其中 {filled_count} 来自单页补齐), "
                f"{events_count} 事件, 共 {pages_count + events_count} 条向量"
            )
            return result
        finally:
            build_lock.release()

    async def _build_embeddings_from_pages(
        self,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> Dict:
        """从页面分析构建向量（降级方案）"""
        page_nums = await self.storage.list_pages()

        if not page_nums:
            logger.warning("没有已分析的页面，跳过构建嵌入")
            return {"success": False, "error": "没有已分析的页面"}

        logger.info(f"从页面分析构建向量: 共 {len(page_nums)} 页")
        self._emit_progress(progress_callback, 0, len(page_nums), "embedding_pages", f"构建页面向量 0/{len(page_nums)}")

        # 兜底路径也要重建 collections，避免保留旧维度或旧事件残留
        if not await self.vector_store.delete_all():
            logger.error("重置向量 collections 失败，终止重建")
            return {"success": False, "error": "重置向量库失败"}

        pages_count = 0
        skip_count = 0

        for page_index, page_num in enumerate(tqdm(page_nums, desc="构建向量嵌入", unit="页"), start=1):
            analysis = await self.storage.load_page_analysis(page_num)
            if not analysis:
                skip_count += 1
                self._emit_progress(progress_callback, page_index, len(page_nums), "embedding_pages", f"构建页面向量 {page_index}/{len(page_nums)}")
                continue

            summary = analysis.get("page_summary", "")
            if summary:
                try:
                    embedding = await self.embedding.embed(summary)
                    added = await self.vector_store.add_page_embedding(
                        page_num, embedding, {
                            "page_summary": summary,
                            "type": "page"
                        }
                    )
                    if added:
                        pages_count += 1
                    else:
                        logger.warning(f"第 {page_num} 页向量写入失败")
                except Exception as e:
                    logger.warning(f"第 {page_num} 页向量化失败: {e}")
                    skip_count += 1
            else:
                skip_count += 1
            self._emit_progress(progress_callback, page_index, len(page_nums), "embedding_pages", f"构建页面向量 {page_index}/{len(page_nums)}")

        result = {
            "success": pages_count > 0,
            "pages_count": pages_count,
            "events_count": 0,
            "total_count": pages_count,
            "skip_count": skip_count
        }

        logger.info(f"向量嵌入构建完成: 成功 {pages_count} 页, 跳过 {skip_count} 页")
        return result
