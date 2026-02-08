"""
Manga Insight 重排序模型客户端

对向量检索结果进行二次排序。
"""

import logging
from typing import List, Dict, Optional

from .clients import get_rerank_url, get_default_model, BaseAPIClient
from .config_models import RerankerConfig

logger = logging.getLogger("MangaInsight.Reranker")


class RerankerClient(BaseAPIClient):
    """
    重排序模型客户端（继承 BaseAPIClient）

    继承自 BaseAPIClient，自动获得：
    - RPM 限制（默认 60 RPM）
    - 指数退避重试（最多 2 次）
    - 本地服务检测和代理禁用
    """

    def __init__(self, config: RerankerConfig):
        """
        初始化 RerankerClient

        Args:
            config: RerankerConfig 配置对象
        """
        self.config = config
        provider = config.provider.lower() if isinstance(config.provider, str) else config.provider.value

        # 获取 rerank 专用 URL
        rerank_url = get_rerank_url(provider, config.base_url)
        self.model = config.model or get_default_model(provider, "reranker")

        # 从完整 URL 中提取 base_url（去掉 /rerank 后缀）
        # 因为 BaseAPIClient 会自动拼接 endpoint
        if rerank_url.endswith("/rerank"):
            base_url = rerank_url[:-7]  # 去掉 "/rerank"
        else:
            base_url = rerank_url.rsplit("/", 1)[0] if "/" in rerank_url else rerank_url

        # 保存完整的 rerank URL 供后续使用
        self._rerank_url = rerank_url

        # 调用父类初始化
        super().__init__(
            provider=provider,
            api_key=config.api_key,
            base_url=base_url,
            rpm_limit=config.rpm_limit if hasattr(config, 'rpm_limit') and config.rpm_limit else 60,
            timeout=30.0,
            max_retries=2
        )

        logger.info(f"RerankerClient 初始化: provider={provider}, rerank_url={self._rerank_url}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回数量

        Returns:
            List[Dict]: 重排序后的文档列表
        """
        if not documents:
            return []

        if not self.config.enabled or not self.config.api_key or not self._rerank_url:
            return documents[:top_k] if top_k else documents

        top_k = top_k or self.config.top_k

        # 提取文档文本
        doc_texts = []
        for doc in documents:
            if isinstance(doc, dict):
                # 尝试不同的字段名
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
            # 使用父类的 _call_api 方法（带重试和 RPM 限制）
            # 但由于 rerank API 端点格式特殊，这里直接使用完整 URL
            await self._enforce_rpm_limit()

            response = await self.client.post(
                self._rerank_url,
                headers=self._get_headers(),
                json={
                    "model": self.model,
                    "query": query,
                    "documents": doc_texts,
                    "top_n": min(top_k, len(documents))
                }
            )
            response.raise_for_status()
            result = response.json()

            # 按重排序结果重新排列文档
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
            # 降级返回原始结果
            return documents[:top_k]

    async def test_connection(self) -> bool:
        """测试连接"""
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
