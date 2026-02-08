"""
Manga Insight API 客户端基类

提供统一的 RPM 限制、重试、流式处理等基础功能。
"""

import asyncio
import json
import logging
import random
import time
from typing import Dict, Optional, Set

import httpx

from .provider_registry import get_base_url

logger = logging.getLogger("MangaInsight.BaseClient")


# 可重试的 HTTP 状态码
RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 504}

# 可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
    httpx.ReadError,
    ConnectionResetError,
)


class RPMLimiter:
    """
    RPM (Requests Per Minute) 限制器

    统一的请求频率控制，避免触发 API 限流。
    """

    def __init__(self, rpm_limit: int = 0):
        """
        初始化 RPM 限制器

        Args:
            rpm_limit: 每分钟最大请求数，0 或负数表示不限制
        """
        self.rpm_limit = rpm_limit
        self._last_reset = 0.0
        self._count = 0

    async def wait(self):
        """
        等待直到可以发送请求

        如果已达到 RPM 限制，会阻塞等待直到下一分钟。
        """
        if self.rpm_limit <= 0:
            return

        current_time = time.time()

        # 重置计数器（每分钟）
        if current_time - self._last_reset >= 60:
            self._last_reset = current_time
            self._count = 0

        # 达到限制，等待
        if self._count >= self.rpm_limit:
            wait_time = 60 - (current_time - self._last_reset)
            if wait_time > 0:
                logger.info(f"RPM 限制: 等待 {wait_time:.1f} 秒")
                await asyncio.sleep(wait_time)
                self._last_reset = time.time()
                self._count = 0

        self._count += 1

    def reset(self):
        """重置计数器"""
        self._last_reset = 0.0
        self._count = 0


class BaseAPIClient:
    """
    API 客户端基类

    提供统一的 HTTP 客户端管理、RPM 限制、流式处理等功能。
    子类只需关注业务逻辑。
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        rpm_limit: int = 0,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """
        初始化基础客户端

        Args:
            provider: 服务商名称
            api_key: API 密钥
            base_url: 自定义 base_url（仅 custom 服务商需要）
            rpm_limit: RPM 限制
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self._base_url = get_base_url(provider, base_url)
        self._rpm_limiter = RPMLimiter(rpm_limit)
        self._timeout = timeout
        self._max_retries = max_retries

        # 创建 HTTP 客户端
        self.client = self._create_http_client()

    def _create_http_client(self) -> httpx.AsyncClient:
        """
        创建 HTTP 客户端

        根据 base_url 判断是否为本地服务，本地服务禁用代理。
        """
        from src.shared.openai_helpers import is_local_service

        if is_local_service(self._base_url):
            logger.info(f"检测到本地服务 ({self._base_url})，禁用代理")
            return httpx.AsyncClient(timeout=self._timeout, trust_env=False)
        return httpx.AsyncClient(timeout=self._timeout)

    @property
    def base_url(self) -> str:
        """获取 base_url"""
        return self._base_url

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    async def __aenter__(self):
        """上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        await self.close()

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _enforce_rpm_limit(self):
        """执行 RPM 限制"""
        await self._rpm_limiter.wait()

    async def _call_api(
        self,
        endpoint: str,
        body: Dict,
        method: str = "POST"
    ) -> Dict:
        """
        调用 API（非流式，带重试）

        Args:
            endpoint: API 端点（如 /chat/completions）
            body: 请求体
            method: HTTP 方法

        Returns:
            Dict: 响应 JSON

        Raises:
            Exception: 重试耗尽后抛出最后的异常
        """
        await self._enforce_rpm_limit()

        url = f"{self._base_url.rstrip('/')}{endpoint}"
        headers = self._get_headers()

        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body
                )

                # 检查是否需要重试
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < self._max_retries:
                        wait_time = self._calculate_backoff(attempt, response)
                        logger.warning(
                            f"API 返回 {response.status_code}，{wait_time:.1f}s 后重试 "
                            f"({attempt + 1}/{self._max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                if response.status_code != 200:
                    error_text = response.text[:500] if response.text else "无响应内容"
                    raise Exception(f"API 错误 {response.status_code}: {error_text}")

                return response.json()

            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self._max_retries:
                    wait_time = self._calculate_backoff(attempt)
                    logger.warning(
                        f"请求失败 ({type(e).__name__})，{wait_time:.1f}s 后重试 "
                        f"({attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

        # 重试耗尽
        if last_exception:
            raise last_exception
        raise Exception("重试耗尽")

    def _calculate_backoff(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None
    ) -> float:
        """
        计算退避时间（指数退避 + 抖动）

        Args:
            attempt: 当前重试次数（0-based）
            response: HTTP 响应（可选，用于读取 Retry-After）

        Returns:
            float: 等待秒数
        """
        # 尝试从 Retry-After 头获取等待时间
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # 指数退避: 2^attempt * (1 + random jitter)
        base_delay = 2 ** attempt
        jitter = random.uniform(0, 0.5)
        return min(base_delay * (1 + jitter), 60.0)  # 最大 60 秒

    async def _call_api_stream(
        self,
        endpoint: str,
        body: Dict,
        print_output: bool = True
    ) -> str:
        """
        调用 API（流式）

        Args:
            endpoint: API 端点
            body: 请求体
            print_output: 是否实时打印输出

        Returns:
            str: 完整响应文本
        """
        await self._enforce_rpm_limit()

        url = f"{self._base_url.rstrip('/')}{endpoint}"
        headers = self._get_headers()
        body["stream"] = True

        full_text = ""
        chunk_count = 0
        model_name = body.get("model", "unknown")

        if print_output:
            print(f"\n[流式输出] {model_name}: ", end="", flush=True)

        async with self.client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                error_bytes = await response.aread()
                error_text = error_bytes.decode("utf-8", errors="ignore")[:500]
                raise Exception(f"API 错误 {response.status_code}: {error_text}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            chunk_count += 1
                            chunk_text = delta["content"]
                            full_text += chunk_text
                            if print_output:
                                print(chunk_text, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

        if print_output:
            print(f"\n[完成] 共 {chunk_count} 块, {len(full_text)} 字符\n")

        return full_text

    async def _call_chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        use_stream: bool = True,
        force_json: bool = False,
        **kwargs
    ) -> str:
        """
        调用聊天补全 API

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            use_stream: 是否使用流式
            force_json: 是否强制 JSON 输出
            **kwargs: 其他参数

        Returns:
            str: 模型响应内容
        """
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if force_json:
            body["response_format"] = {"type": "json_object"}

        if use_stream:
            return await self._call_api_stream("/chat/completions", body)
        else:
            response = await self._call_api("/chat/completions", body)
            choices = response.get("choices", [])
            if not choices:
                raise Exception("API 返回空 choices")
            return choices[0]["message"]["content"]
