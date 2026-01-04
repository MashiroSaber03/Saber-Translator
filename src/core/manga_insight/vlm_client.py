"""
Manga Insight VLM 客户端

统一使用 OpenAI 兼容格式调用多模态大模型。
"""

import base64
import json
import logging
import asyncio
import re
import time
import io
from typing import List, Dict, Optional

import httpx
from PIL import Image

from .config_models import (
    VLMConfig,
    PromptsConfig,
    DEFAULT_BATCH_ANALYSIS_PROMPT
)

logger = logging.getLogger("MangaInsight.VLM")


def resize_image_if_needed(image_bytes: bytes, max_size: int) -> bytes:
    """
    等比例缩放图片，确保最大边不超过 max_size
    
    Args:
        image_bytes: 原始图片字节数据
        max_size: 最大边长（像素），0 或负数表示不压缩
    
    Returns:
        bytes: 处理后的图片字节数据
    """
    if max_size <= 0:
        return image_bytes
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # 如果图片已经足够小，直接返回
        if max(width, height) <= max_size:
            return image_bytes
        
        # 计算等比例缩放比例
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        logger.debug(f"压缩图片: {width}x{height} -> {new_width}x{new_height}")
        
        # 使用高质量缩放算法
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为 JPEG 格式输出（压缩率更高）
        output = io.BytesIO()
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=85)
        
        compressed_bytes = output.getvalue()
        original_size = len(image_bytes) / 1024
        compressed_size = len(compressed_bytes) / 1024
        logger.debug(f"图片大小: {original_size:.1f}KB -> {compressed_size:.1f}KB")
        
        return compressed_bytes
    except Exception as e:
        logger.warning(f"图片压缩失败，使用原图: {e}")
        return image_bytes


class VLMClient:
    """多模态大模型客户端（统一 OpenAI 格式）"""
    
    # 预设服务商的 base_url（全部为 OpenAI 兼容格式）
    PROVIDER_CONFIGS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": ["qwen-vl-max", "qwen-vl-plus"]
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "models": []
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"]
        },
        "volcano": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "models": []
        }
    }
    
    def __init__(self, config: VLMConfig, prompts_config: Optional[PromptsConfig] = None):
        self.config = config
        self.prompts_config = prompts_config or PromptsConfig()
        
        # 检测是否为本地服务（使用共享函数）
        from src.shared.openai_helpers import is_local_service
        # 修复：只有 custom 服务商才使用自定义 URL
        provider_lower = config.provider.lower()
        if provider_lower == 'custom':
            base_url = config.base_url or ""
        else:
            base_url = self.PROVIDER_CONFIGS.get(provider_lower, {}).get("base_url", "")
        
        # 批量分析需要更长超时时间（5分钟）
        if is_local_service(base_url):
            # 本地服务禁用代理
            self.client = httpx.AsyncClient(timeout=300.0, trust_env=False)
            logger.info(f"VLM客户端: 检测到本地服务 ({base_url})，禁用代理")
        else:
            self.client = httpx.AsyncClient(timeout=300.0)
        
        self._rpm_last_reset = 0
        self._rpm_count = 0
    
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.config.api_key and self.config.model)
    
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
                logger.info(f"RPM 限制: 等待 {wait_time:.1f} 秒")
                await asyncio.sleep(wait_time)
                self._rpm_last_reset = time.time()
                self._rpm_count = 0
        
        self._rpm_count += 1
    
    async def analyze_batch(
        self,
        images: List[bytes],
        start_page: int,
        context: Optional[Dict] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        批量分析多个页面（新的四层级模式）
        
        Args:
            images: 图片列表
            start_page: 起始页码
            context: 上下文信息
            custom_prompt: 自定义提示词
        
        Returns:
            Dict: 批量分析结果，包含 pages, batch_summary, key_events 等
        """
        end_page = start_page + len(images) - 1
        prompt = custom_prompt or self._build_batch_analysis_prompt(start_page, end_page, len(images), context)
        
        # 重试循环（包含 JSON 解析失败的情况）
        for attempt in range(self.config.max_retries + 1):
            response_text = await self._call_vlm(
                images=images,
                prompt=prompt
            )
            
            result = self._parse_batch_analysis(response_text, start_page, end_page)
            
            # 如果解析成功（无 parse_error），直接返回
            if not result.get("parse_error"):
                return result
            
            # JSON 解析失败，决定是否重试
            if attempt < self.config.max_retries:
                logger.warning(f"第{start_page}-{end_page}页 JSON 解析失败，第 {attempt + 1} 次重试...")
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"第{start_page}-{end_page}页 JSON 解析失败，已达最大重试次数 ({self.config.max_retries})")
        
        return result
    
    def _build_batch_analysis_prompt(self, start_page: int, end_page: int, page_count: int, context: Dict = None) -> str:
        """构建批量分析提示词"""
        # 优先使用用户自定义提示词，否则使用默认
        base_prompt = self.prompts_config.batch_analysis if self.prompts_config.batch_analysis else DEFAULT_BATCH_ANALYSIS_PROMPT
        # 使用 replace 而不是 format，避免 JSON 示例中的 {} 被误解析
        prompt = base_prompt.replace("{page_count}", str(page_count))
        prompt = prompt.replace("{start_page}", str(start_page))
        prompt = prompt.replace("{end_page}", str(end_page))
        
        if context:
            if context.get("previous_summary"):
                batch_count = context.get("context_batch_count", 1)
                if batch_count > 1:
                    prompt += f"\n\n【前文概要（前{batch_count}批内容）】\n请参考以下前文信息，确保剧情连贯：\n{context['previous_summary']}"
                else:
                    prompt += f"\n\n【前文概要】\n{context['previous_summary']}"
        
        return prompt
    
    async def _call_vlm(self, images: List[bytes], prompt: str) -> str:
        """调用 VLM API（统一使用 OpenAI 格式）"""
        await self._enforce_rpm_limit()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._call_openai_compatible(images, prompt)
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                # 尝试获取更详细的错误信息
                if hasattr(e, 'response'):
                    try:
                        resp = e.response
                        if hasattr(resp, 'text'):
                            error_msg = f"{error_msg} - Response: {resp.text[:500]}"
                        elif hasattr(resp, 'content'):
                            error_msg = f"{error_msg} - Content: {resp.content[:500]}"
                    except:
                        pass
                logger.warning(f"VLM 调用失败 (尝试 {attempt + 1}/{self.config.max_retries + 1}): {error_msg}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise Exception(f"VLM 调用失败: {error_msg}")
    
    async def _call_openai_normal(self, url: str, headers: dict, request_body: dict, provider: str) -> str:
        """OpenAI 兼容 API 非流式调用"""
        response = await self.client.post(url, headers=headers, json=request_body)
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "无响应内容"
            raise Exception(f"{provider} API 错误 {response.status_code}: {error_text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def _call_openai_stream(self, url: str, headers: dict, request_body: dict) -> str:
        """OpenAI 兼容 API 流式调用"""
        request_body["stream"] = True
        full_text = ""
        chunk_count = 0
        model_name = request_body.get('model', 'unknown')
        print(f"\n[流式输出] {model_name}: ", end="", flush=True)
        
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
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            chunk_count += 1
                            chunk_text = delta["content"]
                            full_text += chunk_text
                            # 实时打印到终端
                            print(chunk_text, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n[完成] 共 {chunk_count} 块, {len(full_text)} 字符\n")
        return full_text
    
    async def _call_openai_compatible(self, images: List[bytes], prompt: str) -> str:
        """调用 OpenAI 兼容 API（统一格式，支持所有服务商）"""
        provider = self.config.provider.lower()
        # 修复：只有 custom 服务商才使用自定义 URL
        if provider == 'custom':
            base_url = self.config.base_url or ""
        else:
            base_url = self.PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")
        
        content = []
        for img in images:
            # 根据配置压缩图片以减少 Token 消耗
            img = resize_image_if_needed(img, self.config.image_max_size)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                }
            })
        content.append({"type": "text", "text": prompt})
        
        request_body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature
        }
        
        # 强制 JSON 输出
        if self.config.force_json:
            request_body["response_format"] = {"type": "json_object"}
        
        if not base_url:
            raise ValueError(f"服务商 '{provider}' 需要设置 base_url")
        
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        if self.config.use_stream:
            return await self._call_openai_stream(url, headers, request_body)
        else:
            return await self._call_openai_normal(url, headers, request_body, provider)
    
    def _clean_thinking_tags(self, text: str) -> str:
        """清理思考模型的 thinking 标签"""
        # 移除各种思考标签: <think>, <reasoning>, <thought>, <reflection> 等
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reasoning>.*?</reasoning>',
            r'<thought>.*?</thought>',
            r'<reflection>.*?</reflection>',
            r'<内心独白>.*?</内心独白>',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()
    
    def _extract_json_from_text(self, text: str) -> str:
        """从文本中提取 JSON 内容"""
        # 先清理思考标签
        text = self._clean_thinking_tags(text)
        text = text.strip()
        
        # 移除 markdown 代码块标记
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # 如果还不是以 { 或 [ 开头，尝试查找 JSON
        if not text.startswith('{') and not text.startswith('['):
            # 尝试找到第一个 { 或 [
            json_start = -1
            for i, char in enumerate(text):
                if char in '{[':
                    json_start = i
                    break
            if json_start >= 0:
                text = text[json_start:]
        
        # 尝试找到匹配的结束括号
        text = self._find_complete_json(text)
        
        return text
    
    def _find_complete_json(self, text: str) -> str:
        """找到完整的 JSON 对象或数组"""
        if not text:
            return text
        
        open_char = text[0] if text else ''
        if open_char == '{':
            close_char = '}'
        elif open_char == '[':
            close_char = ']'
        else:
            return text
        
        # 计算括号匹配
        depth = 0
        in_string = False
        escape = False
        
        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"' and not escape:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[:i+1]
        
        return text
    
    def _try_fix_json(self, text: str) -> str:
        """尝试修复常见的 JSON 格式问题"""
        # 移除尾部逗号
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # 修复单引号为双引号（简单情况）
        # 注意：这可能在某些情况下产生问题，所以只在解析失败时尝试
        
        return text
    
    def _parse_batch_analysis(self, response_text: str, start_page: int, end_page: int) -> Dict:
        """解析批量分析结果"""
        text = self._extract_json_from_text(response_text)
        result = None
        
        # 第一次尝试直接解析
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # 第二次尝试：修复常见问题后解析
            try:
                fixed_text = self._try_fix_json(text)
                result = json.loads(fixed_text)
                logger.info(f"第{start_page}-{end_page}页批量 JSON 修复后解析成功")
            except json.JSONDecodeError:
                pass
        
        # 解析失败，返回错误结果
        if result is None:
            logger.warning(f"批量 JSON 解析失败，第{start_page}-{end_page}页")
            page_count = end_page - start_page + 1
            return {
                "page_range": {"start": start_page, "end": end_page},
                "pages": [{
                    "page_number": start_page + i,
                    "raw_response": response_text[:2000] if i == 0 else "",
                    "parse_error": True
                } for i in range(page_count)],
                "batch_summary": "",
                "key_events": [],
                "continuity_notes": "",
                "parse_error": True
            }
        
        # 解析成功，继续处理
        try:
            
            # 确保结果包含必要字段
            if isinstance(result, list):
                # 旧格式兼容：数组转为新格式
                result = {
                    "page_range": {"start": start_page, "end": end_page},
                    "pages": result,
                    "batch_summary": "",
                    "key_events": [],
                    "continuity_notes": ""
                }
            
            # 确保 page_range 存在
            if "page_range" not in result:
                result["page_range"] = {"start": start_page, "end": end_page}
            
            # 确保 pages 字段存在且不为空
            if "pages" not in result or not result["pages"]:
                logger.warning(f"批量分析结果缺少或空的 pages 字段，返回的键: {list(result.keys())}")
                # 尝试从其他可能的键中提取页面数据
                for key in ["page_analyses", "analysis", "results", "data", "page_list"]:
                    if key in result and isinstance(result[key], list) and len(result[key]) > 0:
                        result["pages"] = result[key]
                        logger.info(f"从 '{key}' 字段提取到 {len(result['pages'])} 个页面")
                        break
                else:
                    # 仍然没有数据，根据 batch_summary 自动生成基本页面数据
                    if not result.get("pages"):
                        batch_summary = result.get("batch_summary", "")
                        if batch_summary:
                            # 有批次摘要但无页面详情，为每页生成基本数据
                            logger.info(f"使用 batch_summary 为第{start_page}-{end_page}页生成基本页面数据")
                            result["pages"] = []
                            for page_num in range(start_page, end_page + 1):
                                result["pages"].append({
                                    "page_number": page_num,
                                    "page_summary": batch_summary if page_num == start_page else f"（见第{start_page}页批次摘要）",
                                    "from_batch_summary": True
                                })
                        else:
                            result["pages"] = []
                            logger.warning(f"无法提取页面数据，原始响应前500字符: {response_text[:500]}")
            
            return result
        except Exception as e:
            # 处理其他可能的异常
            logger.warning(f"批量分析结果处理异常: {e}")
            return result if result else {
                "page_range": {"start": start_page, "end": end_page},
                "pages": [],
                "batch_summary": "",
                "key_events": [],
                "continuity_notes": "",
                "parse_error": True
            }
    
    async def test_connection(self) -> bool:
        """测试连接（统一使用 OpenAI 格式）"""
        try:
            test_prompt = "请回复'连接成功'"
            provider = self.config.provider.lower()
            # 修复：只有 custom 服务商才使用自定义 URL
            if provider == 'custom':
                base_url = self.config.base_url or ""
            else:
                base_url = self.PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")
            
            if not base_url:
                logger.error(f"服务商 '{provider}' 未配置 base_url")
                return False
            
            response = await self.client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 10
                }
            )
            
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
