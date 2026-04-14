"""
Manga Insight 角色卡数据模型

面向 SillyTavern V2 角色卡与扩展区结构定义。
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CharacterCandidate:
    """可生成角色卡的候选角色。"""
    name: str
    aliases: List[str] = field(default_factory=list)
    first_appearance: int = 0
    description: str = ""
    arc: str = ""
    dialogue_count: int = 0
    has_dialogues: bool = False
    sample_pages: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterBookEntry:
    """角色卡内嵌世界书条目。"""
    uid: int
    key: List[str]
    name: str = ""
    keysecondary: List[str] = field(default_factory=list)
    comment: str = ""
    content: str = ""
    constant: bool = False
    selective: bool = False
    insertion_order: int = 100
    enabled: bool = True
    position: str = "before_char"
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name or self.comment or f"entry_{self.uid}"
        return {
            "uid": self.uid,
            "id": self.uid,
            "name": name,
            "key": self.key,
            "keys": self.key,
            "keysecondary": self.keysecondary,
            "secondary_keys": self.keysecondary,
            "comment": self.comment,
            "content": self.content,
            "constant": self.constant,
            "selective": self.selective,
            "insertion_order": self.insertion_order,
            "priority": self.insertion_order,
            "enabled": self.enabled,
            "position": self.position,
            "case_sensitive": False,
            "extensions": self.extensions,
        }


@dataclass
class CharacterBookSchema:
    """角色卡内嵌世界书结构。"""
    name: str
    description: str = ""
    scan_depth: int = 2
    token_budget: int = 768
    recursive_scanning: bool = True
    extensions: Dict[str, Any] = field(default_factory=dict)
    entries: List[CharacterBookEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scan_depth": self.scan_depth,
            "token_budget": self.token_budget,
            "recursive_scanning": self.recursive_scanning,
            "extensions": self.extensions,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass
class RegexProfile:
    """扩展区正则模板。"""
    id: str
    name: str
    enabled: bool = True
    scope: str = "character"
    source: str = "ai_output"
    pattern: str = ""
    replacement: str = ""
    flags: str = "g"
    depth_min: int = 0
    depth_max: int = 99
    order: int = 100
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MvuVariable:
    """扩展区 MVU 变量模板。"""
    name: str
    type: str
    scope: str
    default: Any
    value: Any
    validator: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HelperUiManifest:
    """扩展区 Tavern Helper UI 模板。"""
    layout: str = "character-dashboard"
    theme: str = "manga-insight"
    panels: List[Dict[str, Any]] = field(default_factory=list)
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    bindings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterCardV2Draft:
    """单角色角色卡草稿。"""
    character: str
    card: Dict[str, Any]
    source_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character": self.character,
            "card": self.card,
            "source_stats": self.source_stats,
        }


@dataclass
class CompileResult:
    """编译校验结果。"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compiled_cards: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compatibility_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "compiled_cards": self.compiled_cards,
            "compatibility_reports": self.compatibility_reports,
        }


@dataclass
class PngExportResult:
    """PNG 导出结果。"""
    character: str
    file_name: str
    file_path: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
