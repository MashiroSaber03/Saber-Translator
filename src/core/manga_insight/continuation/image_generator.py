"""
Manga Insight 图片生成器

负责调用图片生成API创建漫画页面。
"""

import logging
import os
from typing import Dict, List, Any
from datetime import datetime

from ..config_models import ImageGenConfig
from ..config_utils import load_insight_config
from ..storage import AnalysisStorage
from ..clients import ImageGenClient
from .models import PageContent, ContinuationCharacters
from .reference_tokens import (
    build_continuation_reference_candidates,
    build_original_reference_candidates,
    list_original_manga_page_paths,
    resolve_reference_tokens,
    select_recent_style_reference_tokens,
)

logger = logging.getLogger("MangaInsight.Continuation.ImageGenerator")


class ImageGenerator:
    """图片生成器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.config = load_insight_config()
        self.image_gen_config: ImageGenConfig = self.config.image_gen
        self.storage = AnalysisStorage(book_id)
        self._client = ImageGenClient(self.image_gen_config)
        
        # 生成图片保存目录
        self.output_dir = os.path.join(
            self.storage.base_path,
            "continuation"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    async def close(self) -> None:
        """关闭内部客户端。"""
        await self._client.close()
    
    async def generate_page_image(
        self,
        page_content: PageContent,
        characters: ContinuationCharacters,
        style_reference_tokens: List[str] = None,
        session_id: str = "",
        style_ref_count: int = 3  # 用户设置的画风参考图数量
    ) -> str:
        """
        生成单页漫画图片
        
        Args:
            page_content: 页面内容（包含生图提示词）
            characters: 角色参考图配置
            style_reference_tokens: 画风参考 token 列表
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
        reference_images: List[Dict[str, Any]] = []
        
        # 构建形态映射 {角色名: form_id}
        form_map = {}
        if page_content.character_forms:
            for cf in page_content.character_forms:
                form_map[cf.get("character", "")] = {
                    "form_id": cf.get("form_id"),
                    "form_name": cf.get("form_name"),
                }
        
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
            form_info = form_map.get(char_name, {})
            form = char_ref.resolve_form(
                str(form_info.get("form_id") or ""),
                str(form_info.get("form_name") or ""),
            )
            
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
        
        reference_images.extend(
            await self._resolve_style_reference_images(
                page_content=page_content,
                style_reference_tokens=style_reference_tokens or [],
                style_ref_count=style_ref_count,
            )
        )
        
        # 调用生图 API
        image_data = await self._client.generate(
            full_prompt,
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
        
        image_data = await self._client.generate(
            prompt,
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
        style_reference_tokens: List[str] = None,
        session_id: str = "",
        style_ref_count: int = 3  # 用户设置的画风参考图数量
    ) -> str:
        """
        重新生成页面图片（会保留上一版本）
        
        Args:
            page_content: 页面内容
            characters: 角色参考图配置
            style_reference_tokens: 画风参考 token
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
            style_reference_tokens=style_reference_tokens,
            session_id=session_id,
            style_ref_count=style_ref_count
        )
        
        return new_image_path
    
    def _build_full_prompt(
        self,
        page_content: PageContent
    ) -> str:
        """构建完整的生图提示词

        先声明高优先级的风格规则，再给出本页内容，
        避免详细内容描述反过来压过参考图风格。
        """
        style_rules = """你是一位专业的漫画家。先严格遵守以下风格要求，再根据后面的页面内容作画。

# 风格要求（优先级高于页面内容）
1. 严格沿用原作漫画的线条、上色、角色五官比例、页面密度、分镜节奏。
2. 如果提供了角色参考图，角色的外貌、服装、特征必须与参考图保持一致。
3. 所有对话气泡、文字、音效词必须使用简体中文。
4. 如果页面内容与参考图风格冲突，优先服从参考图风格，只保留这一页必须发生的事件。

# 页面内容
"""

        content = page_content.image_prompt.strip()
        suffix = "\n\n请优先保证与参考图的画风一致性和角色稳定性，再表现这一页发生的事件。"

        return style_rules + content + suffix
    
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

    async def _resolve_style_reference_images(
        self,
        page_content: PageContent,
        style_reference_tokens: List[str],
        style_ref_count: int,
    ) -> List[Dict[str, Any]]:
        """解析当前页可用的画风参考图。"""
        style_ref_count = max(1, int(style_ref_count or 1))
        total_original_pages, candidates = await self._load_style_reference_candidates()
        current_absolute_page = total_original_pages + int(page_content.page_number or 0)

        selected_tokens = [str(token).strip() for token in style_reference_tokens or [] if str(token).strip()]
        if not selected_tokens:
            selected_tokens = select_recent_style_reference_tokens(
                candidates,
                style_ref_count,
                current_page_number=current_absolute_page,
            )

        resolved = resolve_reference_tokens(
            selected_tokens,
            candidates,
            current_page_number=current_absolute_page,
        )

        if len(resolved) < style_ref_count:
            fallback_tokens = select_recent_style_reference_tokens(
                candidates,
                style_ref_count,
                current_page_number=current_absolute_page,
            )
            fallback_refs = resolve_reference_tokens(
                fallback_tokens,
                candidates,
                current_page_number=current_absolute_page,
            )
            seen_paths = {ref["path"] for ref in resolved}
            for ref in fallback_refs:
                if ref["path"] in seen_paths:
                    continue
                resolved.append(ref)
                seen_paths.add(ref["path"])
                if len(resolved) >= style_ref_count:
                    break

        return resolved[:style_ref_count]

    async def _load_style_reference_candidates(self) -> tuple[int, List[Dict[str, Any]]]:
        original_pages = list_original_manga_page_paths(self.book_id)
        if not original_pages:
            logger.warning(f"未找到可用原作页面: {self.book_id}")
        total_original_pages = len(original_pages)
        original_candidates = build_original_reference_candidates(original_pages)

        pages_data = await self.storage.load_continuation_pages()
        continuation_candidates = build_continuation_reference_candidates(
            total_original_pages=total_original_pages,
            pages=(pages_data or {}).get("pages", []),
        )

        return total_original_pages, original_candidates + continuation_candidates

    async def get_recent_style_reference_tokens(
        self,
        count: int = 3,
        current_page_number: int | None = None,
    ) -> List[str]:
        """获取最近可用的画风参考图 token。"""
        _, candidates = await self._load_style_reference_candidates()
        return select_recent_style_reference_tokens(
            candidates,
            count,
            current_page_number=current_page_number,
        )


