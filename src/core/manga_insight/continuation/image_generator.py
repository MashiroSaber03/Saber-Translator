"""
Manga Insight 图片生成器

负责调用图片生成API创建漫画页面。
"""

import logging
import json
import os
from typing import Dict, List
from datetime import datetime

from ..config_models import ImageGenConfig
from ..config_utils import load_insight_config
from ..storage import AnalysisStorage
from ..clients import ImageGenClient
from .models import PageContent, ContinuationCharacters

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


