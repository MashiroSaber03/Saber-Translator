"""
网页漫画导入 - 图片下载器

提供并发图片下载功能，支持：
- 并发控制
- 超时和重试
- 自动添加 Referer
- 自定义 Headers/Cookie
"""

import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse
from dataclasses import dataclass

from .image_processor import ImageProcessor

logger = logging.getLogger("WebImport.ImageDownloader")


@dataclass
class DownloadTask:
    """下载任务"""
    index: int
    url: str
    page_number: int


@dataclass
class DownloadedImage:
    """下载结果"""
    index: int
    filename: str
    data_url: str
    size: int
    success: bool
    error: Optional[str] = None


class ImageDownloader:
    """图片下载器"""
    
    def __init__(self, config: dict = None):
        """
        初始化图片下载器
        
        Args:
            config: 下载配置
                - concurrency: 并发数
                - timeout: 超时时间（秒）
                - retries: 重试次数
                - delay: 下载间隔（毫秒）
                - useReferer: 是否自动添加 Referer
                - customCookie: 自定义 Cookie
                - customHeaders: 自定义 Headers (JSON 字符串)
                - bypassProxy: 是否绕过代理
                - imagePreprocess: 图片预处理配置
        """
        self.config = config or {}
        self.concurrency = self.config.get('concurrency', 3)
        self.timeout = self.config.get('timeout', 30)
        self.retries = self.config.get('retries', 3)
        self.delay = self.config.get('delay', 100) / 1000  # 转换为秒
        self.use_referer = self.config.get('useReferer', True)
        self.custom_cookie = self.config.get('customCookie', '')
        self.custom_headers = self._parse_custom_headers(self.config.get('customHeaders', ''))
        self.bypass_proxy = self.config.get('bypassProxy', False)
        
        # 图片预处理器
        preprocess_config = self.config.get('imagePreprocess', {})
        self.processor = ImageProcessor(preprocess_config)
    
    def _parse_custom_headers(self, headers_str: str) -> Dict[str, str]:
        """解析自定义 Headers JSON 字符串"""
        if not headers_str:
            return {}
        try:
            import json
            return json.loads(headers_str)
        except Exception as e:
            logger.warning(f"解析自定义 Headers 失败: {e}")
            return {}
    
    def _build_headers(self, url: str, source_url: str = None) -> Dict[str, str]:
        """
        构建请求头
        
        Args:
            url: 图片 URL
            source_url: 来源页面 URL（用于 Referer）
        
        Returns:
            请求头字典
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        
        # 添加 Referer
        if self.use_referer and source_url:
            parsed = urlparse(source_url)
            headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"
        
        # 添加自定义 Cookie
        if self.custom_cookie:
            headers['Cookie'] = self.custom_cookie
        
        # 合并自定义 Headers
        headers.update(self.custom_headers)
        
        return headers
    
    async def _download_single(
        self,
        client: httpx.AsyncClient,
        task: DownloadTask,
        source_url: str
    ) -> DownloadedImage:
        """
        下载单张图片
        
        Args:
            client: HTTP 客户端
            task: 下载任务
            source_url: 来源页面 URL
        
        Returns:
            下载结果
        """
        headers = self._build_headers(task.url, source_url)
        last_error = None
        
        for attempt in range(self.retries):
            try:
                response = await client.get(task.url, headers=headers)
                response.raise_for_status()
                
                # 获取图片数据
                image_data = response.content
                
                # 从 URL 获取文件名
                parsed = urlparse(task.url)
                path = parsed.path.split('/')[-1] or f"page_{task.page_number}"
                
                # 预处理图片
                processed_data, ext = self.processor.process(image_data, path)
                
                # 生成文件名
                filename = f"page_{task.page_number:03d}.{ext}"
                
                # 转换为 Data URL
                data_url = ImageProcessor.to_data_url(processed_data, ext)
                
                return DownloadedImage(
                    index=task.index,
                    filename=filename,
                    data_url=data_url,
                    size=len(processed_data),
                    success=True
                )
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.retries}): {task.url} - {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.retries}): {task.url} - {last_error}")
            
            # 重试前等待
            if attempt < self.retries - 1:
                await asyncio.sleep(1)
        
        # 所有重试失败
        return DownloadedImage(
            index=task.index,
            filename=f"page_{task.page_number:03d}.failed",
            data_url="",
            size=0,
            success=False,
            error=last_error
        )
    
    async def download_all(
        self,
        pages: List[Dict[str, Any]],
        source_url: str,
        on_progress: Callable[[int, int], None] = None
    ) -> List[DownloadedImage]:
        """
        下载所有图片
        
        Args:
            pages: 页面列表 [{"pageNumber": 1, "imageUrl": "..."}, ...]
            source_url: 来源页面 URL
            on_progress: 进度回调函数 (current, total)
        
        Returns:
            下载结果列表
        """
        # 创建下载任务
        tasks = [
            DownloadTask(
                index=i,
                url=page.get('imageUrl') or page.get('image_url', ''),
                page_number=page.get('pageNumber') or page.get('page_number', i + 1)
            )
            for i, page in enumerate(pages)
        ]
        
        total = len(tasks)
        results: List[DownloadedImage] = [None] * total
        completed = 0
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # 配置 HTTP 客户端
        client_kwargs = {
            'timeout': httpx.Timeout(self.timeout),
            'follow_redirects': True,
            'http2': True
        }
        
        # 绕过代理
        if self.bypass_proxy:
            client_kwargs['proxy'] = None
        
        async def download_with_semaphore(task: DownloadTask):
            nonlocal completed
            async with semaphore:
                result = await self._download_single(client, task, source_url)
                results[task.index] = result
                completed += 1
                
                if on_progress:
                    on_progress(completed, total)
                
                # 下载间隔
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
                
                return result
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            await asyncio.gather(*[download_with_semaphore(task) for task in tasks])
        
        return results
    
    def download_all_sync(
        self,
        pages: List[Dict[str, Any]],
        source_url: str,
        on_progress: Callable[[int, int], None] = None
    ) -> List[DownloadedImage]:
        """
        同步下载所有图片（完全同步实现，不使用 asyncio）
        
        Args:
            pages: 页面列表
            source_url: 来源页面 URL
            on_progress: 进度回调函数
        
        Returns:
            下载结果列表
        """
        # 创建下载任务
        tasks = []
        for i, page in enumerate(pages):
            tasks.append(DownloadTask(
                index=i,
                url=page.get('imageUrl', ''),
                page_number=page.get('pageNumber', i + 1)
            ))
        
        results = []
        total = len(tasks)
        
        # 使用同步 httpx 客户端逐个下载
        with httpx.Client(timeout=self.timeout) as client:
            for i, task in enumerate(tasks):
                result = self._download_single_sync(client, task, source_url)
                results.append(result)
                
                if on_progress:
                    on_progress(i + 1, total)
        
        return results
    
    def _download_single_sync(
        self,
        client: httpx.Client,
        task: DownloadTask,
        source_url: str
    ) -> DownloadedImage:
        """
        同步下载单张图片
        
        Args:
            client: HTTP 客户端
            task: 下载任务
            source_url: 来源页面 URL
        
        Returns:
            下载结果
        """
        headers = self._build_headers(task.url, source_url)
        last_error = None
        
        for attempt in range(self.retries):
            try:
                response = client.get(task.url, headers=headers)
                response.raise_for_status()
                
                # 获取图片数据
                image_data = response.content
                
                # 从 URL 获取文件名
                parsed = urlparse(task.url)
                path = parsed.path.split('/')[-1] or f"page_{task.page_number}"
                
                # 预处理图片
                processed_data, ext = self.processor.process(image_data, path)
                
                # 生成文件名
                filename = f"page_{task.page_number:03d}.{ext}"
                
                # 转换为 Data URL
                data_url = ImageProcessor.to_data_url(processed_data, ext)
                
                return DownloadedImage(
                    index=task.index,
                    filename=filename,
                    data_url=data_url,
                    size=len(processed_data),
                    success=True
                )
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.retries}): {task.url} - {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.retries}): {task.url} - {last_error}")
            
            # 重试前等待
            if attempt < self.retries - 1:
                import time
                time.sleep(1)
        
        # 所有重试失败
        return DownloadedImage(
            index=task.index,
            filename=f"page_{task.page_number:03d}.failed",
            data_url="",
            size=0,
            success=False,
            error=last_error
        )
