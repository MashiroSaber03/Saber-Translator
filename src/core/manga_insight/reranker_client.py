"""
Manga Insight reranker client backed by shared async transport.
"""

import asyncio
import logging
from typing import List, Dict, Optional

from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedRerankRequest

from .clients import get_rerank_url, get_default_model
from .clients.base_client import RPMLimiter
from .config_models import RerankerConfig

logger = logging.getLogger("MangaInsight.Reranker")
DEFAULT_RERANKER_RPM_LIMIT = 60
DEFAULT_RERANKER_TRANSPORT_RETRIES = 10
DEFAULT_RERANKER_BUSINESS_RETRIES = 10


class RerankerBusinessRetryableError(ValueError):
    """仅用于重排序结果级别的可重试错误。"""


class RerankerClient:
    """
    重排序模型客户端（复用共享 async transport）。
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        provider = config.provider.lower() if isinstance(config.provider, str) else config.provider.value

        rerank_url = get_rerank_url(provider, config.base_url)
        self.model = config.model or get_default_model(provider, "reranker")

        if rerank_url.endswith("/rerank"):
            base_url = rerank_url[:-7]
            endpoint = "/rerank"
        else:
            base_url = rerank_url.rsplit("/", 1)[0] if "/" in rerank_url else rerank_url
            endpoint = rerank_url[len(base_url):] if base_url and rerank_url.startswith(base_url) else "/rerank"

        self.provider = provider
        self._base_url = base_url
        self._rerank_url = rerank_url
        self._endpoint = endpoint or "/rerank"
        timeout_value = float(config.timeout_seconds or 0)
        self._timeout = None if timeout_value <= 0 else timeout_value
        transport_retries = config.transport_retries if config.transport_retries is not None else DEFAULT_RERANKER_TRANSPORT_RETRIES
        business_retries = config.business_retries if config.business_retries is not None else DEFAULT_RERANKER_BUSINESS_RETRIES
        self._rpm_limiter = RPMLimiter(
            DEFAULT_RERANKER_RPM_LIMIT,
            bucket_id=f"rerank:{provider}",
        )
        self._transport = AsyncOpenAICompatibleTransport(
            max_retries=max(0, int(transport_retries))
        )
        self._business_retries = max(0, int(business_retries))

        logger.info(f"RerankerClient 初始化: provider={provider}, rerank_url={self._rerank_url}")

    async def close(self):
        return None

    async def _enforce_rpm_limit(self):
        await self._rpm_limiter.wait()

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if not documents:
            return []

        if not self.config.api_key or not self._rerank_url:
            return documents[:top_k] if top_k else documents

        top_k = top_k or self.config.top_k

        doc_texts = self._build_doc_texts(documents)

        try:
            result = await self._request_rerank_result(query, doc_texts, documents, top_k)
            return self._map_reranked_documents(result, documents, top_k)

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]

    async def test_connection(self) -> bool:
        try:
            result = await self._request_rerank_result(
                query="测试",
                doc_texts=["文档1", "文档2"],
                original_documents=["文档1", "文档2"],
                top_k=2
            )
            mapped = self._map_reranked_documents(result, ["文档1", "文档2"], 2)
            return len(mapped) > 0
        except Exception as e:
            logger.error(f"Reranker 连接测试失败: {e}")
            return False

    @staticmethod
    def _build_doc_texts(documents: List[Dict]) -> List[str]:
        doc_texts: List[str] = []
        for doc in documents:
            if isinstance(doc, dict):
                text = (
                    doc.get("page_summary") or
                    doc.get("document") or
                    doc.get("text") or
                    doc.get("translated_text") or
                    doc.get("content") or
                    str(doc)
                )
                doc_texts.append(text)
            else:
                doc_texts.append(str(doc))
        return doc_texts

    async def _request_rerank_result(
        self,
        query: str,
        doc_texts: List[str],
        original_documents: List[Dict],
        top_k: int,
    ) -> Dict:
        last_error: Optional[Exception] = None
        total_attempts = self._business_retries + 1

        for attempt in range(total_attempts):
            await self._enforce_rpm_limit()
            try:
                result = await self._transport.rerank(
                    UnifiedRerankRequest(
                        provider=self.provider,
                        api_key=self.config.api_key,
                        model=self.model,
                        query=query,
                        documents=doc_texts,
                        top_n=min(top_k, len(original_documents)),
                        base_url=self._base_url or None,
                        timeout=self._timeout,
                        endpoint=self._endpoint,
                    )
                )
                self._validate_rerank_result(result, original_documents)
                return result
            except RerankerBusinessRetryableError as exc:
                last_error = exc
                if attempt >= total_attempts - 1:
                    break
                logger.warning(
                    "Reranker 业务重试 %s/%s: %s",
                    attempt + 1,
                    self._business_retries,
                    exc,
                )
                await asyncio.sleep(1)

        if last_error:
            raise last_error
        raise RuntimeError("重排序结果为空")

    @staticmethod
    def _validate_rerank_result(result: Dict, documents: List[Dict]) -> None:
        results = result.get("results")
        if not isinstance(results, list):
            raise RerankerBusinessRetryableError("Reranker 响应缺少 results 列表")
        if not results:
            raise RerankerBusinessRetryableError("Reranker 响应为空结果")

        valid_count = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < len(documents):
                valid_count += 1

        if valid_count == 0:
            raise RerankerBusinessRetryableError("Reranker 响应没有可映射的结果")

    @staticmethod
    def _map_reranked_documents(result: Dict, documents: List[Dict], top_k: int) -> List[Dict]:
        reranked = []
        for item in result.get("results", []):
            if not isinstance(item, dict):
                continue
            idx = item.get("index", 0)
            if isinstance(idx, int) and idx < len(documents):
                doc = documents[idx].copy() if isinstance(documents[idx], dict) else {"content": documents[idx]}
                doc["rerank_score"] = item.get("relevance_score", 0)
                reranked.append(doc)
        return reranked[:top_k]
