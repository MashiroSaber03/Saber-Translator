"""
Manga Insight Embedding 客户端

支持多种向量模型服务商。
"""

import logging
from typing import List, Optional

from .clients import BaseAPIClient
from .config_models import EmbeddingConfig, ChatLLMConfig

logger = logging.getLogger("MangaInsight.Embedding")


class EmbeddingClient(BaseAPIClient):
    """
    向量模型客户端（继承 BaseAPIClient）

    继承自 BaseAPIClient，自动获得：
    - RPM 限制
    - 指数退避重试
    - 本地服务检测和代理禁用
    """

    def __init__(self, config: EmbeddingConfig):
        """
        初始化 EmbeddingClient

        Args:
            config: EmbeddingConfig 配置对象
        """
        self.config = config

        # 调用父类初始化
        super().__init__(
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            rpm_limit=config.rpm_limit,
            timeout=60.0,
            max_retries=config.max_retries
        )

        logger.info(f"EmbeddingClient 初始化: provider={config.provider}, base_url={self._base_url}")

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

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")

        # 使用父类的 _call_api 方法（带 RPM 限制和重试）
        response = await self._call_api(
            endpoint="/embeddings",
            body={
                "model": self.config.model,
                "input": texts
            }
        )

        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings

    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            embedding = await self.embed("测试文本")
            return len(embedding) > 0
        except Exception as e:
            logger.error(f"Embedding 连接测试失败: {e}")
            return False


class ChatClient(BaseAPIClient):
    """
    对话模型客户端（继承 BaseAPIClient）

    继承自 BaseAPIClient，自动获得：
    - RPM 限制（默认 30 RPM）
    - 指数退避重试（最多 3 次）
    - 流式 SSE 解析
    - 本地服务检测和代理禁用
    """

    def __init__(self, config: ChatLLMConfig):
        """
        初始化 ChatClient

        Args:
            config: ChatLLMConfig 配置对象
        """
        self.config = config

        # 获取 provider 和 base_url
        provider = config.provider.lower() if hasattr(config, 'provider') else "openai"
        custom_url = config.base_url if hasattr(config, 'base_url') and config.base_url else None

        # 调用父类初始化
        super().__init__(
            provider=provider,
            api_key=config.api_key,
            base_url=custom_url,
            rpm_limit=30,  # 添加 RPM 限制（之前缺失）
            timeout=120.0,
            max_retries=3
        )

        logger.info(f"ChatClient 初始化: provider={provider}, base_url={self._base_url}")

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
        logger.debug(f"[ChatClient] provider={self.config.provider}, base_url={self._base_url}, model={self.config.model}")

        if not self._base_url:
            raise ValueError(f"服务商 '{self.config.provider}' 需要设置 base_url")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 检查是否使用流式请求（默认启用，避免超时）
        use_stream = getattr(self.config, 'use_stream', True)
        logger.debug(f"[ChatClient] use_stream={use_stream}, config_type={type(self.config).__name__}")

        # 使用父类的 _call_chat_completion 方法
        return await self._call_chat_completion(
            messages=messages,
            model=self.config.model,
            temperature=temperature,
            use_stream=use_stream
        )

    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 简单测试：发送一个短消息
            response = await self.generate("测试", temperature=0)
            return len(response) > 0
        except Exception as e:
            logger.error(f"LLM 连接测试失败: {e}")
            return False

