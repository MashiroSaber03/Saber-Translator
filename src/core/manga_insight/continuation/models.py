"""
Manga Insight 续写功能数据模型

定义续写功能所需的所有数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class ChapterScript:
    """全话脚本"""
    chapter_title: str
    page_count: int
    script_text: str           # 纯文本脚本（用户可编辑）
    generated_at: str = ""
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_title": self.chapter_title,
            "page_count": self.page_count,
            "script_text": self.script_text,
            "generated_at": self.generated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterScript":
        return cls(
            chapter_title=data.get("chapter_title", ""),
            page_count=data.get("page_count", 0),
            script_text=data.get("script_text", ""),
            generated_at=data.get("generated_at", "")
        )


@dataclass
class PageContent:
    """每页剧情"""
    page_number: int
    characters: List[str]      # 出场人物（向后兼容）
    description: str           # 画面描述
    dialogues: List[Dict]      # 对话列表 [{"character": "角色名", "text": "对话内容"}]
    mood: str                  # 情绪/氛围
    scene: str = ""            # 场景（已废弃，保留用于兼容旧数据）
    image_prompt: str = ""     # 生图提示词（可编辑）
    image_url: str = ""        # 生成的图片URL
    previous_url: str = ""     # 上一版本的图片URL（用于回退）
    status: str = "pending"    # 状态: pending, generating, generated, failed
    # 新增：角色形态列表 [{"character": "角色名", "form_id": "形态ID"}]
    character_forms: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "scene": self.scene,
            "characters": self.characters,
            "character_forms": self.character_forms,
            "description": self.description,
            "dialogues": self.dialogues,
            "mood": self.mood,
            "image_prompt": self.image_prompt,
            "image_url": self.image_url,
            "previous_url": self.previous_url,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageContent":
        return cls(
            page_number=data.get("page_number", 0),
            scene=data.get("scene", ""),
            characters=data.get("characters", []),
            character_forms=data.get("character_forms", []),
            description=data.get("description", ""),
            dialogues=data.get("dialogues", []),
            mood=data.get("mood", ""),
            image_prompt=data.get("image_prompt", ""),
            image_url=data.get("image_url", ""),
            previous_url=data.get("previous_url", ""),
            status=data.get("status", "pending")
        )


@dataclass
class CharacterForm:
    """角色形态"""
    form_id: str              # 形态ID（如 "default", "battle", "dark"）
    form_name: str            # 形态显示名（如 "常服", "战斗服"）
    description: str = ""     # 形态描述
    reference_image: str = "" # 参考图路径
    enabled: bool = True      # 是否启用此形态
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_id": self.form_id,
            "form_name": self.form_name,
            "description": self.description,
            "reference_image": self.reference_image,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterForm":
        return cls(
            form_id=data.get("form_id", ""),
            form_name=data.get("form_name", ""),
            description=data.get("description", ""),
            reference_image=data.get("reference_image", ""),
            enabled=data.get("enabled", True)
        )


@dataclass
class CharacterProfile:
    """角色档案（支持多形态）"""
    name: str                              # 角色名
    aliases: List[str]                     # 别名列表
    description: str                       # 角色基础描述
    forms: List[CharacterForm] = field(default_factory=list)  # 形态列表（可为空）
    enabled: bool = True                   # 是否启用此角色
    
    # 注意：不再自动创建默认形态，forms 列表可以为空
    
    def get_form(self, form_id: str) -> Optional[CharacterForm]:
        """获取指定形态"""
        for form in self.forms:
            if form.form_id == form_id:
                return form
        return None
    
    def get_first_enabled_form(self) -> Optional[CharacterForm]:
        """获取第一个启用的形态（用于回退）"""
        for form in self.forms:
            if form.enabled != False:
                return form
        return None
    
    def get_any_reference_image(self) -> str:
        """获取任意一张参考图（优先返回启用形态的参考图）"""
        # 优先查找启用形态的参考图
        for form in self.forms:
            if form.enabled != False and form.reference_image:
                return form.reference_image
        # 如果没有，返回任意形态的参考图（用于显示）
        for form in self.forms:
            if form.reference_image:
                return form.reference_image
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "forms": [f.to_dict() for f in self.forms],
            "enabled": self.enabled,
            # 向后兼容：提供 reference_image 字段（取任意一张参考图）
            "reference_image": self.get_any_reference_image()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        # 支持从旧格式迁移
        forms_data = data.get("forms", [])
        if not forms_data and data.get("reference_image"):
            # 旧格式：只有单张参考图，创建一个形态
            forms_data = [{
                "form_id": "normal",
                "form_name": "常规",
                "description": "",
                "reference_image": data.get("reference_image", "")
            }]
        
        return cls(
            name=data.get("name", ""),
            aliases=data.get("aliases", []),
            description=data.get("description", ""),
            forms=[CharacterForm.from_dict(f) for f in forms_data],
            enabled=data.get("enabled", True)
        )


# 向后兼容：保留 CharacterReference 别名
CharacterReference = CharacterProfile


@dataclass
class ContinuationCharacters:
    """续写角色配置"""
    book_id: str
    characters: List[CharacterProfile] = field(default_factory=list)
    
    def get_character(self, name: str) -> Optional[CharacterProfile]:
        """获取指定角色信息"""
        for char in self.characters:
            if char.name == name or name in char.aliases:
                return char
        return None
    
    def get_enabled_characters(self) -> List[CharacterProfile]:
        """获取所有已启用的角色"""
        return [char for char in self.characters if char.enabled != False]
    
    def get_enabled_forms(self, character_name: str) -> List[CharacterForm]:
        """获取指定角色的所有已启用形态"""
        char = self.get_character(character_name)
        if not char:
            return []
        return [form for form in char.forms if form.enabled != False]


    
    def get_character_image(self, name: str) -> Optional[str]:
        """获取指定角色的任意一张参考图路径（向后兼容）"""
        char = self.get_character(name)
        return char.get_any_reference_image() if char else None
    
    def get_form_reference(self, name: str, form_id: str) -> Optional[str]:
        """获取指定角色指定形态的参考图"""
        char = self.get_character(name)
        if not char:
            return None
        form = char.get_form(form_id)
        if form:
            return form.reference_image
        # 不再回退到默认形态，如果指定形态不存在则返回None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "characters": [c.to_dict() for c in self.characters]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContinuationCharacters":
        return cls(
            book_id=data.get("book_id", ""),
            characters=[CharacterProfile.from_dict(c) for c in data.get("characters", [])]
        )
