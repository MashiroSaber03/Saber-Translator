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
        
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        request_body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature
        }
        
        # 检查是否使用流式请求（默认启用，避免超时）
        use_stream = getattr(self.config, 'use_stream', True)
        logger.info(f"[ChatClient] use_stream={use_stream}, config_type={type(self.config).__name__}")
        
        if use_stream:
            return await self._generate_stream(url, headers, request_body)
        else:
            return await self._generate_normal(url, headers, request_body)
    
    async def _generate_normal(self, url: str, headers: dict, request_body: dict) -> str:
        """普通请求"""
        response = await self.client.post(url, headers=headers, json=request_body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def _generate_stream(self, url: str, headers: dict, request_body: dict) -> str:
        """流式请求（避免超时）"""
        import json as json_module
        
        request_body["stream"] = True
        full_text = ""
        model_name = request_body.get('model', 'unknown')
        logger.info(f"[ChatClient 流式请求] 模型: {model_name}")
        
        async with self.client.stream("POST", url, headers=headers, json=request_body) as response:
            if response.status_code != 200:
                error_bytes = await response.aread()
                error_text = error_bytes.decode('utf-8', errors='ignore')[:500]
                raise Exception(f"API 错误 {response.status_code}: {error_text}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json_module.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            full_text += delta["content"]
                    except json_module.JSONDecodeError:
                        continue
        
        logger.info(f"[ChatClient 流式完成] {len(full_text)} 字符")
        return full_text
