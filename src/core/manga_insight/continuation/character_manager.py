"""
Manga Insight 角色管理器

管理续写功能的角色参考图。
"""

import logging
import os
import json
import shutil
from typing import Dict, List, Optional, Any
from datetime import datetime

from PIL import Image

from ..storage import AnalysisStorage
from .models import CharacterProfile, CharacterForm, ContinuationCharacters

# 向后兼容别名
CharacterReference = CharacterProfile

logger = logging.getLogger("MangaInsight.Continuation.CharacterManager")


class CharacterManager:
    """角色参考图管理器"""
    
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.storage = AnalysisStorage(book_id)
        
        # 角色图片保存目录
        self.characters_dir = os.path.join(
            self.storage.base_path,
            "continuation",
            "characters"
        )
        os.makedirs(self.characters_dir, exist_ok=True)
        
        # 配置文件路径
        self.config_path = os.path.join(
            self.storage.base_path,
            "continuation",
            "characters.json"
        )
    
    async def load_characters_from_timeline(self) -> List[CharacterProfile]:
        """
        从时间线数据加载角色信息
        
        Returns:
            List[CharacterProfile]: 角色列表（不含形态和参考图，由用户添加）
        """
        timeline_data = await self.storage.load_timeline()
        
        if not timeline_data:
            logger.warning("时间线数据不存在")
            return []
        
        characters = []
        
        for char_data in timeline_data.get("characters", []):
            # 创建角色（不自动创建形态，由用户添加）
            char_profile = CharacterProfile(
                name=char_data.get("name", ""),
                aliases=char_data.get("aliases", []),
                description=char_data.get("description", ""),
                forms=[]  # 空的形态列表
            )
            characters.append(char_profile)
        
        logger.info(f"从时间线加载了 {len(characters)} 个角色")
        return characters
    
    def load_characters(self) -> ContinuationCharacters:
        """
        加载已保存的角色配置
        
        Returns:
            ContinuationCharacters: 角色配置
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ContinuationCharacters.from_dict(data)
            except Exception as e:
                logger.error(f"加载角色配置失败: {e}")
        
        # 返回空配置
        return ContinuationCharacters(
            book_id=self.book_id,
            characters=[]
        )
    
    def save_characters(self, characters: ContinuationCharacters) -> bool:
        """
        保存角色配置
        
        Args:
            characters: 角色配置
            
        Returns:
            bool: 是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(characters.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"角色配置已保存: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"保存角色配置失败: {e}")
            return False
    
    async def initialize_characters(self) -> ContinuationCharacters:
        """
        初始化角色配置（从时间线加载并创建配置）
        
        Returns:
            ContinuationCharacters: 初始化后的角色配置
        """
        # 先尝试加载已保存的配置
        existing = self.load_characters()
        if existing.characters:
            logger.info("使用已保存的角色配置")
            return existing
        
        # 从时间线加载
        char_refs = await self.load_characters_from_timeline()
        
        characters = ContinuationCharacters(
            book_id=self.book_id,
            characters=char_refs
        )
        
        # 保存配置
        self.save_characters(characters)
        
        return characters
    
    def update_character_reference(
        self,
        character_name: str,
        reference_image: str,
        form_id: str
    ) -> bool:
        """
        更新角色指定形态的参考图
        
        Args:
            character_name: 角色名
            reference_image: 参考图路径
            form_id: 形态ID
            
        Returns:
            bool: 是否更新成功
        """
        characters = self.load_characters()
        
        # 查找角色
        char = characters.get_character(character_name)
        
        if char:
            # 查找形态
            form = char.get_form(form_id)
            if not form:
                # 形态不存在，无法设置参考图
                logger.warning(f"形态不存在: {character_name}/{form_id}")
                return False
            
            # 删除旧图片（如果存在且不同）
            if form.reference_image and form.reference_image != reference_image:
                if os.path.exists(form.reference_image):
                    try:
                        os.remove(form.reference_image)
                    except Exception as e:
                        logger.warning(f"删除旧图片失败: {e}")
            
            form.reference_image = reference_image
            return self.save_characters(characters)
        else:
            # 添加新角色（不创建默认形态，创建用户指定的形态）
            new_form = CharacterForm(
                form_id=form_id,
                form_name=form_id,  # 使用 form_id 作为初始名
                description="",
                reference_image=reference_image
            )
            characters.characters.append(CharacterProfile(
                name=character_name,
                aliases=[],
                description="",
                forms=[new_form]
            ))
        
        # 保存配置
        return self.save_characters(characters)
    
    def get_character_descriptions(self) -> Dict[str, str]:
        """
        获取所有角色的描述字典
        
        Returns:
            Dict[str, str]: {角色名: 描述}
        """
        characters = self.load_characters()
        
        return {
            char.name: char.description or f"角色 {char.name}"
            for char in characters.characters
        }
    
    
    # ===== 形态管理方法 =====
    
    def add_form(
        self,
        character_name: str,
        form_id: str,
        form_name: str,
        description: str = ""
    ) -> Optional[CharacterForm]:
        """
        为角色添加新形态
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            form_name: 形态显示名
            description: 形态描述
            
        Returns:
            CharacterForm: 新创建的形态，失败返回 None
        """
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return None
        
        # 检查形态ID是否已存在
        if char.get_form(form_id):
            logger.warning(f"形态已存在: {character_name}/{form_id}")
            return None
        
        # 创建新形态
        new_form = CharacterForm(
            form_id=form_id,
            form_name=form_name,
            description=description,
            reference_image=""
        )
        
        char.forms.append(new_form)
        self.save_characters(characters)
        
        logger.info(f"已添加形态: {character_name}/{form_id}")
        return new_form
    
    def update_form(
        self,
        character_name: str,
        form_id: str,
        form_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        更新角色形态信息
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            form_name: 新的形态名（可选）
            description: 新的描述（可选）
            
        Returns:
            bool: 是否更新成功
        """
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return False
        
        form = char.get_form(form_id)
        if not form:
            logger.warning(f"形态不存在: {character_name}/{form_id}")
            return False
        
        if form_name is not None:
            form.form_name = form_name
        if description is not None:
            form.description = description
        
        self.save_characters(characters)
        logger.info(f"已更新形态: {character_name}/{form_id}")
        return True
    
    def toggle_character_enabled(
        self,
        character_name: str,
        enabled: bool
    ) -> bool:
        """
        切换角色的启用状态
        
        Args:
            character_name: 角色名
            enabled: 是否启用
            
        Returns:
            bool: 是否更新成功
        """
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return False
        
        char.enabled = enabled
        
        self.save_characters(characters)
        logger.info(f"已{'启用' if enabled else '禁用'}角色: {character_name}")
        return True
    
    def toggle_form_enabled(
        self,
        character_name: str,
        form_id: str,
        enabled: bool
    ) -> bool:
        """
        切换角色形态的启用状态
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            enabled: 是否启用
            
        Returns:
            bool: 是否更新成功
        """
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return False
        
        form = char.get_form(form_id)
        if not form:
            logger.warning(f"形态不存在: {character_name}/{form_id}")
            return False
        
        form.enabled = enabled
        
        self.save_characters(characters)
        logger.info(f"已{'启用' if enabled else '禁用'}形态: {character_name}/{form_id}")
        return True
    

    def delete_form(self, character_name: str, form_id: str) -> bool:
        """
        删除角色形态
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            
        Returns:
            bool: 是否删除成功
        """
        # 所有形态都可以删除，不再有默认形态的限制
        
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return False
        
        # 查找并删除形态
        for i, form in enumerate(char.forms):
            if form.form_id == form_id:
                # 删除参考图
                if form.reference_image and os.path.exists(form.reference_image):
                    try:
                        os.remove(form.reference_image)
                        logger.info(f"已删除形态参考图: {form.reference_image}")
                    except Exception as e:
                        logger.warning(f"删除形态参考图失败: {e}")
                
                char.forms.pop(i)
                self.save_characters(characters)
                logger.info(f"已删除形态: {character_name}/{form_id}")
                return True
        
        logger.warning(f"形态不存在: {character_name}/{form_id}")
        return False
    
    def upload_form_image(
        self,
        character_name: str,
        form_id: str,
        image_path: str,
        is_temp: bool = True
    ) -> Optional[str]:
        """
        为角色形态上传参考图
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            image_path: 图片路径
            is_temp: 是否是临时文件
            
        Returns:
            str: 保存后的图片路径，失败返回 None
        """
        characters = self.load_characters()
        
        char = characters.get_character(character_name)
        if not char:
            logger.warning(f"角色不存在: {character_name}")
            return None
        
        form = char.get_form(form_id)
        if not form:
            logger.warning(f"形态不存在: {character_name}/{form_id}")
            return None
        
        # 验证图片文件
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 生成保存路径（格式：角色名_形态ID_时间戳.扩展名）
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ext = os.path.splitext(image_path)[1].lower() or ".png"
        save_filename = f"{character_name}_{form_id}_{timestamp}{ext}"
        save_path = os.path.join(self.characters_dir, save_filename)
        
        # 删除旧图片
        if form.reference_image and os.path.exists(form.reference_image):
            try:
                os.remove(form.reference_image)
            except Exception as e:
                logger.warning(f"删除旧图片失败: {e}")
        
        # 复制或移动文件
        if is_temp:
            shutil.copy2(image_path, save_path)
        else:
            shutil.move(image_path, save_path)
        
        # 更新配置
        form.reference_image = save_path
        self.save_characters(characters)
        
        logger.info(f"已上传形态参考图: {character_name}/{form_id} -> {save_path}")
        return save_path
    
    def get_form_reference_image(
        self,
        character_name: str,
        form_id: str
    ) -> Optional[str]:
        """
        获取指定角色指定形态的参考图路径
        
        Args:
            character_name: 角色名
            form_id: 形态ID
            
        Returns:
            str: 参考图路径，不存在返回 None
        """
        characters = self.load_characters()
        return characters.get_form_reference(character_name, form_id)
    
    def get_character_forms_tree(self, include_disabled: bool = False) -> Dict[str, Any]:
        """
        获取角色形态树状结构（用于提示词生成）
        
        Args:
            include_disabled: 是否包含禁用的角色和形态
        
        Returns:
            Dict: 树状结构的角色形态信息
        """
        characters = self.load_characters()
        
        tree = {}
        for char in characters.characters:
            # 跳过禁用的角色（除非指定包含）
            if not include_disabled and char.enabled == False:
                continue
            
            tree[char.name] = {
                "description": char.description or f"角色 {char.name}",
                "aliases": char.aliases,
                "enabled": char.enabled,
                "forms": {}
            }
            for form in char.forms:
                # 跳过禁用的形态（除非指定包含）
                if not include_disabled and form.enabled == False:
                    continue
                
                tree[char.name]["forms"][form.form_id] = {
                    "form_name": form.form_name,
                    "description": form.description or "",
                    "has_image": bool(form.reference_image and os.path.exists(form.reference_image)),
                    "enabled": form.enabled
                }
        
        return tree
    
    def format_characters_for_prompt(self) -> str:
        """
        格式化角色信息为提示词文本（树状结构）
        
        只包含启用的角色和形态
        
        Returns:
            str: 格式化后的角色信息
        """
        tree = self.get_character_forms_tree(include_disabled=False)
        
        if not tree:
            return "（暂无角色信息）"
        
        lines = ["【角色档案】"]
        for char_name, char_info in tree.items():
            lines.append(f"├── {char_name}")
            lines.append(f"│   ├── 描述: {char_info['description']}")
            if char_info['aliases']:
                lines.append(f"│   ├── 别名: {', '.join(char_info['aliases'])}")
            lines.append(f"│   └── 形态:")
            
            forms = char_info['forms']
            if not forms:
                lines.append("│       └── （无可用形态）")
                continue
            
            form_items = list(forms.items())
            for i, (form_id, form_info) in enumerate(form_items):
                prefix = "│       └──" if i == len(form_items) - 1 else "│       ├──"
                has_img = "✓有参考图" if form_info['has_image'] else "✗无参考图"
                desc = f": {form_info['description']}" if form_info['description'] else ""
                lines.append(f"{prefix} {form_id} ({form_info['form_name']}) [{has_img}]{desc}")
        
        return "\n".join(lines)

