"""
Manga Insight Embedding 客户端

支持多种向量模型服务商。
"""

import logging
import asyncio
import time
from typing import List, Optional

import httpx

from .config_models import EmbeddingConfig

logger = logging.getLogger("MangaInsight.Embedding")


class EmbeddingClient:
    """向量模型客户端（统一 OpenAI 格式）"""
    
    # 预设服务商的 base_url（全部为 OpenAI 兼容格式）
    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "text-embedding-3-small"
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "default_model": "text-embedding-004"
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "text-embedding-v3"
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "BAAI/bge-m3"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat"
        },
        "volcano": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "default_model": ""
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
        self._rpm_last_reset = 0
        self._rpm_count = 0
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
    
    async def _enforce_rpm_limit(self):
        """执行 RPM 限制"""
        if self.config.rpm_limit <= 0:
            return
        
        current_time = time.time()
        
        if current_time - self._rpm_last_reset >= 60:
            self._rpm_last_reset = current_time
            self._rpm_count = 0
        
        if self._rpm_count >= self.config.rpm_limit:
            wait_time = 60 - (current_time - self._rpm_last_reset)
            if wait_time > 0:
                logger.info(f"Embedding RPM 限制: 等待 {wait_time:.1f} 秒")
                await asyncio.sleep(wait_time)
                self._rpm_last_reset = time.time()
                self._rpm_count = 0
        
        self._rpm_count += 1
    
    async def embed(self, text: str) -> List[float]:
        """
        生成单个文本的向量
        
        Args:
            text: 输入文本
        
        Returns:
            List[float]: 向量
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []
        
        await self._enforce_rpm_limit()
        
        provider = self.config.provider.lower()
        base_url = self.config.base_url or self.PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")
        
        if not base_url:
            raise ValueError(f"服务商 '{provider}' 需要设置 base_url")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json={
                        "model": self.config.model,
                        "input": texts
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding 调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        return []
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            embedding = await self.embed("测试文本")
            return len(embedding) > 0
        except Exception as e:
            logger.error(f"Embedding 连接测试失败: {e}")
            return False


class ChatClient:
    """对话模型客户端（统一 OpenAI 格式）"""
    
    # 预设服务商的 base_url（全部为 OpenAI 兼容格式）
    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1"
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1"
        },
        "volcano": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3"
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        生成回答（统一使用 OpenAI 格式）
        
        Args:
            prompt: 用户提示
            system: 系统提示
            temperature: 温度参数
        
        Returns:
            str: 生成的文本
        """
        provider = self.config.provider.lower() if hasattr(self.config, 'provider') else "openai"
        base_url = self.config.base_url if hasattr(self.config, 'base_url') and self.config.base_url else self.PROVIDER_CONFIGS.get(provider, {}).get("base_url", "https://api.openai.com/v1")
        
        if not base_url:
            raise ValueError(f"服务商 '{provider}' 需要设置 base_url")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
