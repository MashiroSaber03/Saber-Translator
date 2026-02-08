"""
Manga Insight 图片生成器

负责调用图片生成API创建漫画页面。
"""

import logging
import json
import base64
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx
from openai import OpenAI  # OpenAI SDK
from PIL import Image
import io

from ..config_models import ImageGenConfig
from ..config_utils import load_insight_config
from ..storage import AnalysisStorage
from ..clients import get_image_gen_base_url, get_image_gen_url
from .models import PageContent, ContinuationCharacters

logger = logging.getLogger("MangaInsight.Continuation.ImageGenerator")


class ImageGenerator:
    """图片生成器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.config = load_insight_config()
        self.image_gen_config: ImageGenConfig = self.config.image_gen
        self.storage = AnalysisStorage(book_id)
        
        # 生成图片保存目录
        self.output_dir = os.path.join(
            self.storage.base_path,
            "continuation"
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def generate_page_image(
        self,
        page_content: PageContent,
        characters: ContinuationCharacters,
        style_reference_images: List[str] = None,
        session_id: str = "",
        style_ref_count: int = 3  # 用户设置的画风参考图数量
    ) -> str:
        """
        生成单页漫画图片
        
        Args:
            page_content: 页面内容（包含生图提示词）
            characters: 角色参考图配置
            style_reference_images: 画风参考图片路径列表
            session_id: 续写会话ID
            style_ref_count: 画风参考图数量（从用户设置获取）
            
        Returns:
            str: 生成的图片路径
        """
        if not page_content.image_prompt:
            raise ValueError(f"第 {page_content.page_number} 页没有生图提示词")
        
        # 构建完整提示词
        full_prompt = self._build_full_prompt(
            page_content=page_content
        )
        
        # 收集参考图片
        reference_images = []
        
        # 构建形态映射 {角色名: form_id}
        form_map = {}
        if page_content.character_forms:
            for cf in page_content.character_forms:
                form_map[cf.get("character", "")] = cf.get("form_id")  # 可能为 None
        
        # 添加角色参考图
        for char_name in page_content.characters:
            char_ref = characters.get_character(char_name)
            if not char_ref:
                continue
            
            # 检查角色是否启用
            if char_ref.enabled == False:
                logger.debug(f"角色 {char_name} 已禁用，跳过参考图")
                continue
            
            # 获取该角色在此页使用的形态
            form_id = form_map.get(char_name)
            form = char_ref.get_form(form_id) if form_id else None
            
            # 获取形态的参考图路径
            ref_image = None
            form_name = None
            
            # 如果指定形态存在且启用
            if form and form.enabled != False and form.reference_image:
                ref_image = form.reference_image
                form_name = form.form_name
            else:
                # 形态不存在、被禁用或没有参考图，尝试获取第一个启用且有参考图的形态
                for f in char_ref.forms:
                    if f.enabled != False and f.reference_image:
                        ref_image = f.reference_image
                        form_name = f.form_name
                        logger.debug(f"角色 {char_name} 使用回退形态: {form_name}")
                        break
            
            if ref_image:
                # 构建标签：有形态名就显示，没有就只显示角色名
                label = f"{char_name} - {form_name}" if form_name else char_name
                reference_images.append({
                    "path": ref_image,
                    "type": "character",
                    "name": label  # 这个名字会被用作图片标签
                })
        
        # 添加画风参考图（取最后N张，形成滑动窗口）
        if style_reference_images:
            # 使用用户设置的数量，确保取最后 style_ref_count 张
            for img_path in style_reference_images[-style_ref_count:]:
                reference_images.append({
                    "path": img_path,
                    "type": "style"
                })
        
        # 调用生图API
        image_data = await self._call_image_api(
            prompt=full_prompt,
            reference_images=reference_images
        )
        
        # 保存图片
        image_path = self._save_image(
            image_data=image_data,
            page_number=page_content.page_number,
            session_id=session_id
        )
        
        return image_path
    
    async def generate_character_orthographic(
        self,
        character_name: str,
        source_image_paths: list[str],
        form_id: str
    ) -> str:
        """
        生成角色三视图
        
        Args:
            character_name: 角色名（用于保存路径）
            source_image_paths: 原始参考图路径列表
            form_id: 形态ID（用于保存路径）
            
        Returns:
            str: 生成的三视图图片路径
        """
        # 构建提示词
        prompt = self._build_orthographic_prompt()
        
        # 调用生图API，使用所有上传的图片作为参考
        reference_images = []
        for i, img_path in enumerate(source_image_paths):
            reference_images.append({
                "path": img_path,
                "type": "character",
                "name": f"ref_{i+1}"
            })
        
        logger.info(f"使用 {len(reference_images)} 张参考图生成三视图: {character_name}/{form_id}")
        
        image_data = await self._call_image_api(
            prompt=prompt,
            reference_images=reference_images
        )
        
        # 保存三视图到角色目录下
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = os.path.join(
            self.output_dir,
            "characters",
            character_name,
            f"{character_name}_{form_id}_orthographic_{timestamp}.png"
        )
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return image_path
    
    async def regenerate_page_image(
        self,
        page_content: PageContent,
        characters: ContinuationCharacters,
        style_reference_images: List[str] = None,
        session_id: str = "",
        style_ref_count: int = 3  # 用户设置的画风参考图数量
    ) -> str:
        """
        重新生成页面图片（会保留上一版本）
        
        Args:
            page_content: 页面内容
            characters: 角色参考图配置
            style_reference_images: 画风参考图片
            session_id: 会话ID
            style_ref_count: 画风参考图数量（滑动窗口大小）
            
        Returns:
            str: 新生成的图片路径
        """
        # 保存当前版本为上一版本
        if page_content.image_url:
            page_content.previous_url = page_content.image_url
        
        # 生成新图片
        new_image_path = await self.generate_page_image(
            page_content=page_content,
            characters=characters,
            style_reference_images=style_reference_images,
            session_id=session_id,
            style_ref_count=style_ref_count
        )
        
        return new_image_path
    
    def _build_full_prompt(
        self,
        page_content: PageContent
    ) -> str:
        """构建完整的生图提示词
        
        前置：漫画家角色设定
        中间：AI 生成的剧情描述
        后置：统一的规则
        """
        # 前置提示词：设定角色
        prefix = """你是一位专业的漫画家。根据我提供的参考资料，画出下一页漫画。

"""
        
        # 中间：AI 生成的剧情提示词
        content = page_content.image_prompt
        
        # 后置提示词：统一规则
        suffix = """

---

**重要规则**：
1. **画风继承**：画风、线条、配色完全参考提供的原漫画页面
2. **角色外貌**：如果提供了角色参考图，角色的外貌、服装、特征必须与参考图保持一致
3. **文字语言**：所有对话气泡、文字、音效词必须使用**简体中文**"""
        
        return prefix + content + suffix
    
    def _build_orthographic_prompt(self) -> str:
        """构建三视图生成提示词"""
        return """基于我提供的参考图片，生成角色三视图设计稿。

重要要求：
- 必须完全参考我上传的图片中的角色外观
- 保持与参考图一致的：发型、发色、眼睛颜色、服装、体型、装备、特殊特征
- 不要创造新角色，只是将参考图中的角色画成三视图

三视图格式要求：
- 白色背景
- 从左到右依次展示：正面、侧面（3/4视角）、背面
- 三个视角必须是同一个角色的同一形态
- 保持所有视角的设计完全一致
- 日系动漫/漫画风格
- 清晰的线条和配色

请生成这个角色的三视图。"""
    
    async def _call_image_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """
        调用图片生成API
        
        Args:
            prompt: 生图提示词
            reference_images: 参考图片列表
            
        Returns:
            bytes: 生成的图片数据
        """
        config = self.image_gen_config
        
        # 根据服务商选择API格式
        if config.provider == "openai":
            return await self._call_openai_api(prompt, reference_images)
        elif config.provider == "siliconflow":
            return await self._call_siliconflow_api(prompt, reference_images)
        elif config.provider == "qwen":
            return await self._call_qwen_api(prompt, reference_images)
        elif config.provider == "volcano":
            return await self._call_volcano_api(prompt, reference_images)
        else:
            # 默认使用 OpenAI 兼容格式
            return await self._call_openai_compatible_api(prompt, reference_images)
    
    async def _call_openai_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """
        使用 OpenAI SDK 的 chat.completions.create 调用图生图 API
        支持传入参考图片
        """
        config = self.image_gen_config
        
        # 创建 OpenAI 客户端
        base_url = config.base_url or get_image_gen_base_url("openai")
        client = OpenAI(
            api_key=config.api_key,
            base_url=base_url,
        )
        
        # 构建消息内容
        content = []
        
        # 添加参考图片（如果有）
        if reference_images:
            for ref_img in reference_images:
                img_path = ref_img.get("path", "")
                if img_path and os.path.exists(img_path):
                    # 如果是角色参考图，添加角色名标签
                    char_name = ref_img.get("name") if ref_img.get("type") == "character" else None
                    img_b64 = self._encode_image_to_base64(img_path, character_name=char_name)
                    if img_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                        img_type = ref_img.get("type", "unknown")
                        if char_name:
                            logger.info(f"已添加角色参考图: {char_name} ({img_path})")
                        else:
                            logger.info(f"已添加{img_type}参考图: {img_path}")
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        for retry in range(config.max_retries):
            try:
                # 使用 chat.completions.create 方式调用
                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=4096,
                    n=1
                )
                
                # 获取响应
                result = response.choices[0].message.content
                
                # 尝试从响应中提取图片
                if result:
                    # 处理 Markdown 格式: ![image](data:image/jpeg;base64,...)
                    import re
                    md_match = re.search(r'!\[.*?\]\((data:image/[^;]+;base64,([^)]+))\)', result)
                    if md_match:
                        b64_data = md_match.group(2)
                        logger.info("从 Markdown 格式中提取图片数据")
                        return base64.b64decode(b64_data)
                    
                    # 纯 data:image 格式
                    if result.startswith("data:image"):
                        b64_data = result.split(",", 1)[-1]
                        return base64.b64decode(b64_data)
                    
                    # 纯 base64 字符串
                    elif result.startswith("/9j/") or result.startswith("iVBOR"):
                        return base64.b64decode(result)
                    
                    # URL
                    elif result.startswith("http"):
                        async with httpx.AsyncClient(timeout=60) as http_client:
                            img_response = await http_client.get(result)
                            return img_response.content
                    
                    else:
                        logger.warning(f"未知的响应格式: {result[:300]}...")
                        raise ValueError("无法解析图片响应")
                else:
                    raise ValueError("API 响应为空")
                    
            except Exception as e:
                logger.warning(f"OpenAI API 调用失败 (尝试 {retry + 1}/{config.max_retries}): {e}")
                if retry < config.max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise
    
    def _encode_image_to_base64(self, image_path: str, character_name: str = None) -> str:
        """
        将图片编码为 base64
        
        Args:
            image_path: 图片路径
            character_name: 角色名（如果提供，会在图片底部添加名称标签）
            
        Returns:
            str: base64 编码的图片
        """
        try:
            # 如果有角色名，给图片添加标签
            if character_name:
                labeled_image = self._add_character_label(image_path, character_name)
                if labeled_image:
                    return labeled_image
            
            # 普通编码
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败: {e}")
            return ""
    
    def _add_character_label(self, image_path: str, character_name: str) -> str:
        """
        给图片底部添加角色名标签
        
        Args:
            image_path: 原始图片路径
            character_name: 要添加的角色名
            
        Returns:
            str: 带标签的图片的 base64 编码
        """
        try:
            from PIL import ImageDraw, ImageFont
            
            # 打开原图
            img = Image.open(image_path)
            
            # 计算标签区域高度（图片高度的 8%，最小30像素，最大80像素）
            label_height = max(30, min(80, int(img.height * 0.08)))
            
            # 创建新图片（原图 + 白色标签区域）
            new_height = img.height + label_height
            new_img = Image.new('RGB', (img.width, new_height), 'white')
            
            # 粘贴原图到顶部
            new_img.paste(img, (0, 0))
            
            # 准备绘制文字
            draw = ImageDraw.Draw(new_img)
            
            # 尝试加载字体（优先使用系统中文字体）
            font = None
            font_size = int(label_height * 0.6)
            font_paths = [
                # Windows 中文字体
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
                # macOS 中文字体
                "/System/Library/Fonts/PingFang.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                # Linux 中文字体
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            # 如果没有找到字体，使用默认字体
            if font is None:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # 计算文字位置（居中）
            text = character_name
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (img.width - text_width) // 2
            y = img.height + (label_height - text_height) // 2
            
            # 绘制黑色文字
            draw.text((x, y), text, fill='black', font=font)
            
            # 转换为 base64
            buffer = io.BytesIO()
            # 根据原图格式保存
            img_format = 'PNG' if image_path.lower().endswith('.png') else 'JPEG'
            new_img.save(buffer, format=img_format, quality=95)
            buffer.seek(0)
            
            logger.debug(f"已为角色 '{character_name}' 添加标签")
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            logger.warning(f"添加角色标签失败: {e}，将使用原图")
            return ""
    
    async def _call_siliconflow_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """调用 SiliconFlow API"""
        config = self.image_gen_config
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": config.model,
            "prompt": prompt,
            "batch_size": 1,
            "num_inference_steps": 30
        }
        
        # 如果有参考图，添加到请求中
        if reference_images:
            # SiliconFlow 的某些模型支持 image_prompts
            encoded_images = []
            for ref in reference_images[:3]:  # 最多3张
                try:
                    with open(ref["path"], "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                        encoded_images.append({
                            "image": f"data:image/png;base64,{img_data}",
                            "weight": 0.3 if ref["type"] == "style" else 0.5
                        })
                except Exception as e:
                    logger.warning(f"无法读取参考图 {ref['path']}: {e}")
            
            if encoded_images:
                body["image_prompts"] = encoded_images
        
        base_url = config.base_url or get_image_gen_base_url("siliconflow")

        async with httpx.AsyncClient(timeout=300) as client:  # 5分钟超时，图片生成可能很耗时
            for retry in range(config.max_retries):
                try:
                    response = await client.post(
                        f"{base_url}/images/generations",
                        headers=headers,
                        json=body
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # 处理返回格式
                    if "images" in result:
                        image_data = result["images"][0]
                        if image_data.get("url"):
                            # 下载图片
                            img_response = await client.get(image_data["url"])
                            return img_response.content
                        elif image_data.get("b64_json"):
                            return base64.b64decode(image_data["b64_json"])
                    elif "data" in result:
                        image_data = result["data"][0]
                        if image_data.get("url"):
                            img_response = await client.get(image_data["url"])
                            return img_response.content
                        elif image_data.get("b64_json"):
                            return base64.b64decode(image_data["b64_json"])
                    
                    raise ValueError("无法解析 API 响应")
                    
                except Exception as e:
                    logger.warning(f"SiliconFlow API 调用失败 (尝试 {retry + 1}/{config.max_retries}): {e}")
                    if retry < config.max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    else:
                        raise
    
    async def _call_qwen_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """调用通义万相 API"""
        config = self.image_gen_config
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": config.model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "n": 1
            }
        }
        
        base_url = config.base_url or get_image_gen_base_url("qwen")

        async with httpx.AsyncClient(timeout=300) as client:  # 5分钟超时
            for retry in range(config.max_retries):
                try:
                    response = await client.post(
                        f"{base_url}/services/aigc/text2image/image-synthesis",
                        headers=headers,
                        json=body
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # 通义万相是异步API，需要轮询
                    if result.get("output", {}).get("task_status") == "PENDING":
                        task_id = result["output"]["task_id"]
                        return await self._poll_qwen_task(client, task_id, headers, base_url)
                    
                    # 直接返回结果
                    if "output" in result and "results" in result["output"]:
                        image_url = result["output"]["results"][0]["url"]
                        img_response = await client.get(image_url)
                        return img_response.content
                    
                    raise ValueError("无法解析 API 响应")
                    
                except Exception as e:
                    logger.warning(f"通义万相 API 调用失败 (尝试 {retry + 1}/{config.max_retries}): {e}")
                    if retry < config.max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    else:
                        raise
    
    async def _poll_qwen_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        headers: Dict,
        base_url: str,
        max_polls: int = 60
    ) -> bytes:
        """轮询通义万相任务状态"""
        for _ in range(max_polls):
            await asyncio.sleep(2)
            
            response = await client.get(
                f"{base_url}/tasks/{task_id}",
                headers=headers
            )
            result = response.json()
            
            status = result.get("output", {}).get("task_status")
            if status == "SUCCEEDED":
                image_url = result["output"]["results"][0]["url"]
                img_response = await client.get(image_url)
                return img_response.content
            elif status == "FAILED":
                raise ValueError(f"任务失败: {result.get('output', {}).get('message')}")
        
        raise TimeoutError("任务超时")
    
    async def _call_volcano_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """调用火山引擎 API"""
        config = self.image_gen_config
        
        # 火山引擎需要特殊的签名认证，这里简化处理
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": config.model,
            "prompt": prompt,
            "n": 1
        }
        
        base_url = config.base_url or get_image_gen_base_url("volcano")

        async with httpx.AsyncClient(timeout=300) as client:  # 5分钟超时
            for retry in range(config.max_retries):
                try:
                    response = await client.post(
                        f"{base_url}/v1/images/generations",
                        headers=headers,
                        json=body
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    if "data" in result:
                        image_url = result["data"][0].get("url")
                        if image_url:
                            img_response = await client.get(image_url)
                            return img_response.content
                    
                    raise ValueError("无法解析 API 响应")
                    
                except Exception as e:
                    logger.warning(f"火山引擎 API 调用失败 (尝试 {retry + 1}/{config.max_retries}): {e}")
                    if retry < config.max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    else:
                        raise
    
    async def _call_openai_compatible_api(
        self,
        prompt: str,
        reference_images: List[Dict] = None
    ) -> bytes:
        """
        使用 OpenAI SDK 的 chat.completions.create 调用兼容格式的图生图 API
        支持传入参考图片
        """
        config = self.image_gen_config
        
        if not config.base_url:
            raise ValueError("自定义服务商需要设置 base_url")
        
        # 创建 OpenAI 客户端（使用自定义 base_url）
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        
        # 构建消息内容
        content = []
        
        # 添加参考图片（如果有）
        if reference_images:
            for ref_img in reference_images:
                img_path = ref_img.get("path", "")
                if img_path and os.path.exists(img_path):
                    # 如果是角色参考图，添加角色名标签
                    char_name = ref_img.get("name") if ref_img.get("type") == "character" else None
                    img_b64 = self._encode_image_to_base64(img_path, character_name=char_name)
                    if img_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                        img_type = ref_img.get("type", "unknown")
                        if char_name:
                            logger.info(f"已添加角色参考图: {char_name} ({img_path})")
                        else:
                            logger.info(f"已添加{img_type}参考图: {img_path}")
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        for retry in range(config.max_retries):
            try:
                # 使用 chat.completions.create 方式调用
                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=4096,
                    n=1
                )
                
                # 获取响应
                result = response.choices[0].message.content
                
                # 尝试从响应中提取图片
                if result:
                    # 处理 Markdown 格式: ![image](data:image/jpeg;base64,...)
                    import re
                    md_match = re.search(r'!\[.*?\]\((data:image/[^;]+;base64,([^)]+))\)', result)
                    if md_match:
                        b64_data = md_match.group(2)
                        logger.info("从 Markdown 格式中提取图片数据")
                        return base64.b64decode(b64_data)
                    
                    # 纯 data:image 格式
                    if result.startswith("data:image"):
                        b64_data = result.split(",", 1)[-1]
                        return base64.b64decode(b64_data)
                    
                    # 纯 base64 字符串
                    elif result.startswith("/9j/") or result.startswith("iVBOR"):
                        return base64.b64decode(result)
                    
                    # URL
                    elif result.startswith("http"):
                        async with httpx.AsyncClient(timeout=60) as http_client:
                            img_response = await http_client.get(result)
                            return img_response.content
                    
                    else:
                        logger.warning(f"未知的响应格式: {result[:300]}...")
                        raise ValueError("无法解析图片响应")
                else:
                    raise ValueError("API 响应为空")
                    
            except Exception as e:
                logger.warning(f"API 调用失败 (尝试 {retry + 1}/{config.max_retries}): {e}")
                if retry < config.max_retries - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise
    
    def _save_image(
        self,
        image_data: bytes,
        page_number: int,
        session_id: str = ""
    ) -> str:
        """保存生成的图片"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 构建文件名
        if session_id:
            filename = f"{session_id}_page{page_number:03d}_{timestamp}.png"
        else:
            filename = f"page{page_number:03d}_{timestamp}.png"
        
        # 保存路径
        image_path = os.path.join(self.output_dir, filename)
        
        # 保存图片
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"图片已保存: {image_path}")
        return image_path
    
    def get_style_reference_images(
        self,
        count: int = 3
    ) -> List[str]:
        """
        获取原漫画最后N页作为画风参考图
        
        返回的列表按时间从早到晚排序（最老的在前，最新的在后）
        这样前端使用 slice(-N) 滑动窗口时，能正确删除最老的页面
        
        注意：已生成页面的滑动窗口由前端自己维护，这里只返回原漫画的参考图
        
        Args:
            count: 需要的参考图数量
            
        Returns:
            List[str]: 图片路径列表，按时间从早到晚排序
        """
        style_refs = []
        
        # 从原漫画最后几页获取（倒序遍历，收集最新的 count 张）
        book_pages = self._get_original_manga_pages()
        for page_path in reversed(book_pages):
            if os.path.exists(page_path):
                style_refs.append(page_path)
                if len(style_refs) >= count:
                    break
        
        # 反转，使列表按时间从早到晚排序
        # 例如：[page100, page99, page98] -> [page98, page99, page100]
        style_refs.reverse()
        return style_refs
    
    def _get_original_manga_pages(self) -> List[str]:
        """获取原漫画的页面路径"""
        import json
        from src.shared.path_helpers import resource_path
        from src.core import bookshelf_manager
        
        pages = []
        
        try:
            # 从书架系统获取书籍信息
            book = bookshelf_manager.get_book(self.book_id)
            if not book:
                logger.warning(f"未找到书籍: {self.book_id}")
                return pages
            
            # 获取章节信息
            chapters = book.get("chapters", [])

            for chapter in chapters:
                chapter_id = chapter.get("id")
                if not chapter_id:
                    continue

                # 从 session_meta.json 获取图片信息（使用新路径格式）
                session_dir = resource_path(f"data/bookshelf/{self.book_id}/chapters/{chapter_id}/session")
                session_meta_path = os.path.join(session_dir, "session_meta.json")
                
                if os.path.exists(session_meta_path):
                    try:
                        with open(session_meta_path, "r", encoding="utf-8") as f:
                            session_data = json.load(f)
                        
                        # 支持两种格式
                        if "total_pages" in session_data:
                            image_count = session_data.get("total_pages", 0)
                        else:
                            images_meta = session_data.get("images_meta", [])
                            image_count = len(images_meta)
                        
                        for i in range(image_count):
                            image_path = self._find_image_path(session_dir, i)
                            if image_path:
                                pages.append(image_path)
                    except Exception as e:
                        logger.warning(f"读取 session_meta 失败: {session_meta_path}, {e}")
                        continue
        except Exception as e:
            logger.error(f"获取原漫画页面失败: {e}")
        
        return pages
    
    def _find_image_path(self, session_dir: str, image_index: int, image_type: str = "original") -> str:
        """查找图片文件路径"""
        # 格式: images/{idx}/{type}.png
        image_path = os.path.join(session_dir, "images", str(image_index), f"{image_type}.png")
        if os.path.exists(image_path):
            return image_path
        
        return ""


