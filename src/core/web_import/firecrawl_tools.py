"""
网页漫画导入 - Firecrawl 工具定义

定义 AI Agent 可调用的 Firecrawl 工具 (Function Calling Schema)
"""

import logging
import httpx
from typing import Dict, Any, Optional, List

logger = logging.getLogger("WebImport.FirecrawlTools")

# Firecrawl API 基础 URL
FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"


# ============================================================
# 工具定义 (OpenAI Function Calling Schema)
# ============================================================

FIRECRAWL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": "抓取单个网页的内容。可以获取页面的 HTML、Markdown 内容、截图，以及执行页面操作（如滚动、点击、等待）。适用于需要深入分析单个页面的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页 URL"
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["markdown", "html", "screenshot", "links"]},
                        "description": "返回格式列表。可选: markdown, html, screenshot, links",
                        "default": ["markdown", "html"]
                    },
                    "wait_for": {
                        "type": "integer",
                        "description": "等待页面加载的毫秒数（用于动态内容）",
                        "default": 0
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["wait", "click", "scroll", "screenshot"],
                                    "description": "操作类型"
                                },
                                "milliseconds": {
                                    "type": "integer",
                                    "description": "wait 操作的等待时间"
                                },
                                "selector": {
                                    "type": "string", 
                                    "description": "click 操作的 CSS 选择器"
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": ["up", "down"],
                                    "description": "scroll 操作的方向"
                                },
                                "amount": {
                                    "type": "integer",
                                    "description": "scroll 操作的滚动像素数"
                                }
                            }
                        },
                        "description": "页面操作列表，用于处理动态加载的内容"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "firecrawl_extract",
            "description": "使用 LLM 从网页中提取结构化数据。适用于需要从页面中提取特定信息（如漫画图片列表）的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要提取数据的 URL 列表"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "描述要提取什么数据的提示词"
                    },
                    "schema": {
                        "type": "object",
                        "description": "期望的输出 JSON Schema（可选）"
                    }
                },
                "required": ["urls"]
            }
        }
    }
]


class FirecrawlClient:
    """Firecrawl API 客户端"""
    
    def __init__(self, api_key: str, timeout: int = 60):
        """
        初始化 Firecrawl 客户端
        
        Args:
            api_key: Firecrawl API Key
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def scrape(
        self,
        url: str,
        formats: List[str] = None,
        wait_for: int = 0,
        actions: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        抓取单个网页
        
        Args:
            url: 目标 URL
            formats: 返回格式列表
            wait_for: 等待时间（毫秒）
            actions: 页面操作列表
        
        Returns:
            抓取结果
        """
        payload = {"url": url}
        
        if formats:
            payload["formats"] = formats
        if wait_for > 0:
            payload["waitFor"] = wait_for
        if actions:
            payload["actions"] = actions
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{FIRECRAWL_API_BASE}/scrape",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def extract(
        self,
        urls: List[str],
        prompt: str = None,
        schema: Dict = None
    ) -> Dict[str, Any]:
        """
        从网页提取结构化数据
        
        Args:
            urls: URL 列表
            prompt: 提取提示词
            schema: 输出 Schema
        
        Returns:
            提取结果
        """
        payload = {"urls": urls}
        
        if prompt:
            payload["prompt"] = prompt
        if schema:
            payload["schema"] = schema
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{FIRECRAWL_API_BASE}/extract",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()


async def execute_firecrawl_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    api_key: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    执行 Firecrawl 工具调用 (异步版本)
    
    Args:
        tool_name: 工具名称
        tool_args: 工具参数
        api_key: Firecrawl API Key
        timeout: 超时时间
    
    Returns:
        工具执行结果
    """
    client = FirecrawlClient(api_key, timeout)
    
    try:
        if tool_name == "firecrawl_scrape":
            result = await client.scrape(
                url=tool_args.get("url"),
                formats=tool_args.get("formats", ["markdown", "html"]),
                wait_for=tool_args.get("wait_for", 0),
                actions=tool_args.get("actions")
            )
        elif tool_name == "firecrawl_extract":
            result = await client.extract(
                urls=tool_args.get("urls", []),
                prompt=tool_args.get("prompt"),
                schema=tool_args.get("schema")
            )
        else:
            return {"error": f"未知的工具: {tool_name}"}
        
        return result
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Firecrawl API 错误: {e.response.status_code} - {e.response.text}")
        return {"error": f"API 错误: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"执行 Firecrawl 工具失败: {e}")
        return {"error": str(e)}


def execute_firecrawl_tool_sync(
    tool_name: str,
    tool_args: Dict[str, Any],
    api_key: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    执行 Firecrawl 工具调用 (同步版本)
    
    Args:
        tool_name: 工具名称
        tool_args: 工具参数
        api_key: Firecrawl API Key
        timeout: 超时时间
    
    Returns:
        工具执行结果
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        if tool_name == "firecrawl_scrape":
            payload = {"url": tool_args.get("url")}
            formats = tool_args.get("formats", ["markdown", "html"])
            if formats:
                payload["formats"] = formats
            wait_for = tool_args.get("wait_for", 0)
            if wait_for > 0:
                payload["waitFor"] = wait_for
            actions = tool_args.get("actions")
            if actions:
                payload["actions"] = actions
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{FIRECRAWL_API_BASE}/scrape",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
                
        elif tool_name == "firecrawl_extract":
            payload = {"urls": tool_args.get("urls", [])}
            prompt = tool_args.get("prompt")
            if prompt:
                payload["prompt"] = prompt
            schema = tool_args.get("schema")
            if schema:
                payload["schema"] = schema
            
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{FIRECRAWL_API_BASE}/extract",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        else:
            return {"error": f"未知的工具: {tool_name}"}
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Firecrawl API 错误: {e.response.status_code} - {e.response.text}")
        return {"error": f"API 错误: {e.response.status_code}", "details": e.response.text}
    except Exception as e:
        logger.error(f"执行 Firecrawl 工具失败: {e}")
        return {"error": str(e)}
