"""
Manga Insight reranker client backed by shared async transport.
"""

import logging
from typing import List, Dict, Optional

from src.shared.ai_transport import AsyncOpenAICompatibleTransport, UnifiedRerankRequest

from .clients import get_rerank_url, get_default_model
from .clients.base_client import RPMLimiter
from .config_models import RerankerConfig

logger = logging.getLogger("MangaInsight.Reranker")
DEFAULT_RERANKER_RPM_LIMIT = 60
DEFAULT_RERANKER_MAX_RETRIES = 2


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
        self._timeout = 30.0
        self._rpm_limiter = RPMLimiter(DEFAULT_RERANKER_RPM_LIMIT)
        self._transport = AsyncOpenAICompatibleTransport(max_retries=DEFAULT_RERANKER_MAX_RETRIES)

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

        doc_texts = []
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

        try:
            await self._enforce_rpm_limit()
            result = await self._transport.rerank(
                UnifiedRerankRequest(
                    provider=self.provider,
                    api_key=self.config.api_key,
                    model=self.model,
                    query=query,
                    documents=doc_texts,
                    top_n=min(top_k, len(documents)),
                    base_url=self._base_url or None,
                    timeout=self._timeout,
                    endpoint=self._endpoint,
                )
            )

            reranked = []
            for item in result.get("results", []):
                idx = item.get("index", 0)
                if idx < len(documents):
                    doc = documents[idx].copy() if isinstance(documents[idx], dict) else {"content": documents[idx]}
                    doc["rerank_score"] = item.get("relevance_score", 0)
                    reranked.append(doc)

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]

    async def test_connection(self) -> bool:
        try:
            result = await self.rerank(
                query="测试",
                documents=["文档1", "文档2"],
                top_k=2
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Reranker 连接测试失败: {e}")
            return False
