"""
Manga Insight 重排序模型客户端

对向量检索结果进行二次排序。
"""

import logging
from typing import List, Dict, Optional

import httpx

from .config_models import RerankerConfig

logger = logging.getLogger("MangaInsight.Reranker")


class RerankerClient:
    """重排序模型客户端"""
    
    # 预设服务商的 rerank API base_url
    PROVIDER_CONFIGS = {
        "jina": {
            "base_url": "https://api.jina.ai/v1/rerank",
            "default_model": "jina-reranker-v2-base-multilingual"
        },
        "cohere": {
            "base_url": "https://api.cohere.ai/v1/rerank",
            "default_model": "rerank-multilingual-v3.0"
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1/rerank",
            "default_model": "BAAI/bge-reranker-v2-m3"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1/rerank",
            "default_model": ""
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "default_model": ""
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/rerank",
            "default_model": ""
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1/rerank",
            "default_model": ""
        },
        "volcano": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3/rerank",
            "default_model": ""
        }
    }
    
    def __init__(self, config: RerankerConfig):
        self.config = config
        provider = config.provider.lower() if isinstance(config.provider, str) else config.provider.value
        self.provider_config = self.PROVIDER_CONFIGS.get(provider, {})
        # 修复：只有 custom 服务商才使用自定义 URL
        if provider == 'custom':
            self.base_url = config.base_url or ""
        else:
            self.base_url = self.provider_config.get("base_url", "")
        self.model = config.model or self.provider_config.get("default_model", "")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
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
        
        if not self.config.enabled or not self.config.api_key or not self.base_url:
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
            response = await self.client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
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
